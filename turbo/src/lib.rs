use numpy::{PyReadonlyArray4, PyReadwriteArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum ResizeAlgorithm {
    Nearest,
    Bilinear,
    Area,
}

impl ResizeAlgorithm {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            "area" => Ok(Self::Area),
            _ => Err(PyValueError::new_err(
                "algorithm must be 'nearest', 'bilinear', or 'area'",
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct ImagePlan {
    raw_h: usize,
    raw_w: usize,
    source_h: usize,
    source_w: usize,
    out_h: usize,
    out_w: usize,
    out_c: usize,
    crop: [usize; 4],
    mask_crop: bool,
    crop_fill: u8,
    algorithm: ResizeAlgorithm,
}

impl ImagePlan {
    #[inline]
    fn source_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        source_y: usize,
        source_x: usize,
    ) -> [u8; 3] {
        let (raw_y, raw_x) = if self.mask_crop {
            (source_y, source_x)
        } else {
            (source_y + self.crop[0], source_x + self.crop[2])
        };
        if self.mask_crop
            && (raw_y < self.crop[0]
                || raw_y >= self.raw_h - self.crop[1]
                || raw_x < self.crop[2]
                || raw_x >= self.raw_w - self.crop[3])
        {
            return [self.crop_fill; 3];
        }
        let offset = (raw_y * self.raw_w + raw_x) * 3;
        let mut rgb = [current[offset], current[offset + 1], current[offset + 2]];
        if let Some(prior) = previous {
            rgb[0] = rgb[0].max(prior[offset]);
            rgb[1] = rgb[1].max(prior[offset + 1]);
            rgb[2] = rgb[2].max(prior[offset + 2]);
        }
        rgb
    }

    #[inline]
    fn nearest_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let source_y = out_y * self.source_h / self.out_h;
        let source_x = out_x * self.source_w / self.out_w;
        self.source_rgb(current, previous, source_y, source_x)
    }

    #[inline]
    fn bilinear_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let source_y = ((out_y as f64 + 0.5) * self.source_h as f64 / self.out_h as f64 - 0.5)
            .clamp(0.0, (self.source_h - 1) as f64);
        let source_x = ((out_x as f64 + 0.5) * self.source_w as f64 / self.out_w as f64 - 0.5)
            .clamp(0.0, (self.source_w - 1) as f64);
        let y0 = source_y.floor() as usize;
        let x0 = source_x.floor() as usize;
        let y1 = (y0 + 1).min(self.source_h - 1);
        let x1 = (x0 + 1).min(self.source_w - 1);
        let wy = source_y - y0 as f64;
        let wx = source_x - x0 as f64;
        let p00 = self.source_rgb(current, previous, y0, x0);
        let p01 = self.source_rgb(current, previous, y0, x1);
        let p10 = self.source_rgb(current, previous, y1, x0);
        let p11 = self.source_rgb(current, previous, y1, x1);
        let mut result = [0_u8; 3];
        for channel in 0..3 {
            let top = p00[channel] as f64 * (1.0 - wx) + p01[channel] as f64 * wx;
            let bottom = p10[channel] as f64 * (1.0 - wx) + p11[channel] as f64 * wx;
            result[channel] = (top * (1.0 - wy) + bottom * wy).round() as u8;
        }
        result
    }

    #[inline]
    fn area_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let y_start = out_y as f64 * self.source_h as f64 / self.out_h as f64;
        let y_end = (out_y + 1) as f64 * self.source_h as f64 / self.out_h as f64;
        let x_start = out_x as f64 * self.source_w as f64 / self.out_w as f64;
        let x_end = (out_x + 1) as f64 * self.source_w as f64 / self.out_w as f64;
        let mut sums = [0.0_f64; 3];
        let mut weight_sum = 0.0_f64;
        for source_y in y_start.floor() as usize..(y_end.ceil() as usize).min(self.source_h) {
            let y_weight =
                (y_end.min(source_y as f64 + 1.0) - y_start.max(source_y as f64)).max(0.0);
            for source_x in x_start.floor() as usize..(x_end.ceil() as usize).min(self.source_w) {
                let x_weight =
                    (x_end.min(source_x as f64 + 1.0) - x_start.max(source_x as f64)).max(0.0);
                let weight = y_weight * x_weight;
                let rgb = self.source_rgb(current, previous, source_y, source_x);
                for channel in 0..3 {
                    sums[channel] += rgb[channel] as f64 * weight;
                }
                weight_sum += weight;
            }
        }
        [
            (sums[0] / weight_sum).round() as u8,
            (sums[1] / weight_sum).round() as u8,
            (sums[2] / weight_sum).round() as u8,
        ]
    }

    #[inline]
    fn resized_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        match self.algorithm {
            ResizeAlgorithm::Nearest => self.nearest_rgb(current, previous, out_y, out_x),
            ResizeAlgorithm::Bilinear => self.bilinear_rgb(current, previous, out_y, out_x),
            ResizeAlgorithm::Area => self.area_rgb(current, previous, out_y, out_x),
        }
    }

    fn write_frame(&self, current: &[u8], previous: Option<&[u8]>, output: &mut [u8]) {
        for out_y in 0..self.out_h {
            for out_x in 0..self.out_w {
                let rgb = self.resized_rgb(current, previous, out_y, out_x);
                let offset = (out_y * self.out_w + out_x) * self.out_c;
                if self.out_c == 1 {
                    output[offset] = ((u32::from(rgb[0]) * 77
                        + u32::from(rgb[1]) * 150
                        + u32::from(rgb[2]) * 29
                        + 128)
                        >> 8) as u8;
                } else {
                    output[offset..offset + 3].copy_from_slice(&rgb);
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (current, output, crop, mask_crop, crop_fill, algorithm, previous=None))]
#[allow(clippy::too_many_arguments)]
fn preprocess_into(
    py: Python<'_>,
    current: PyReadonlyArray4<'_, u8>,
    mut output: PyReadwriteArray4<'_, u8>,
    crop: Vec<usize>,
    mask_crop: bool,
    crop_fill: u8,
    algorithm: &str,
    previous: Option<PyReadonlyArray4<'_, u8>>,
) -> PyResult<()> {
    let current_shape = current.shape();
    let output_shape = output.shape();
    if current_shape.len() != 4
        || current_shape[3] != 3
        || output_shape.len() != 4
        || output_shape[0] != current_shape[0]
        || !matches!(output_shape[3], 1 | 3)
    {
        return Err(PyValueError::new_err(
            "current must be NHWC RGB and output must be NHWC with one or three channels",
        ));
    }
    if crop.len() != 4 {
        return Err(PyValueError::new_err(
            "crop must contain top, bottom, left, right",
        ));
    }
    let raw_h = current_shape[1];
    let raw_w = current_shape[2];
    if crop[0] + crop[1] >= raw_h || crop[2] + crop[3] >= raw_w {
        return Err(PyValueError::new_err(
            "crop must preserve at least one source pixel",
        ));
    }
    if let Some(prior) = previous.as_ref()
        && prior.shape() != current_shape
    {
        return Err(PyValueError::new_err(
            "previous must have the same shape as current",
        ));
    }
    let plan = ImagePlan {
        raw_h,
        raw_w,
        source_h: if mask_crop {
            raw_h
        } else {
            raw_h - crop[0] - crop[1]
        },
        source_w: if mask_crop {
            raw_w
        } else {
            raw_w - crop[2] - crop[3]
        },
        out_h: output_shape[1],
        out_w: output_shape[2],
        out_c: output_shape[3],
        crop: [crop[0], crop[1], crop[2], crop[3]],
        mask_crop,
        crop_fill,
        algorithm: ResizeAlgorithm::parse(algorithm)?,
    };
    let current_data = current.as_slice()?;
    let previous_data = previous
        .as_ref()
        .map(|value| value.as_slice())
        .transpose()?;
    let output_data = output.as_slice_mut()?;
    let raw_frame_size = raw_h * raw_w * 3;
    let output_frame_size = plan.out_h * plan.out_w * plan.out_c;
    py.detach(|| {
        output_data
            .par_chunks_mut(output_frame_size)
            .zip(current_data.par_chunks(raw_frame_size))
            .enumerate()
            .for_each(|(lane, (output_frame, current_frame))| {
                let prior = previous_data
                    .map(|data| &data[lane * raw_frame_size..(lane + 1) * raw_frame_size]);
                plan.write_frame(current_frame, prior, output_frame);
            });
    });
    Ok(())
}

#[pymodule]
fn _vizdoom_turbo(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocess_into, module)?)?;
    Ok(())
}
