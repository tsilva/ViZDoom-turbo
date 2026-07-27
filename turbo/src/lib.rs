use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray1,
    PyReadwriteArray3, PyReadwriteArray4, PyReadwriteArray5, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

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

const MASKED_SAMPLE: usize = u32::MAX as usize;
const INDEXED_AREA_DIVISOR: u64 = 1_600;
const INDEXED_AREA_WEIGHT_QUANTUM: u32 = 48;
const INDEXED_AREA_CHANNEL_BITS: u32 = 20;
const INDEXED_AREA_CHANNEL_MASK: u64 = (1 << INDEXED_AREA_CHANNEL_BITS) - 1;

struct AreaSample {
    offset: u32,
    weight: u32,
}

struct AreaPixel {
    sample_start: u32,
    sample_end: u32,
}

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
    nearest_y: Vec<usize>,
    nearest_x: Vec<usize>,
    linear_y: Vec<(usize, usize, f64)>,
    linear_x: Vec<(usize, usize, f64)>,
    area_divisor: u64,
    area_samples: Vec<AreaSample>,
    area_pixels: Vec<AreaPixel>,
}

impl ImagePlan {
    #[allow(clippy::too_many_arguments)]
    fn new(
        raw_h: usize,
        raw_w: usize,
        out_h: usize,
        out_w: usize,
        out_c: usize,
        crop: [usize; 4],
        mask_crop: bool,
        crop_fill: u8,
        algorithm: ResizeAlgorithm,
    ) -> Self {
        let source_h = if mask_crop {
            raw_h
        } else {
            raw_h - crop[0] - crop[1]
        };
        let source_w = if mask_crop {
            raw_w
        } else {
            raw_w - crop[2] - crop[3]
        };
        let area_y = Self::area_axis(source_h, out_h);
        let area_x = Self::area_axis(source_w, out_w);
        let mut area_samples = Vec::new();
        let mut area_pixels = Vec::new();
        if matches!(algorithm, ResizeAlgorithm::Area) {
            area_pixels.reserve(out_h * out_w);
            for y_samples in &area_y {
                for x_samples in &area_x {
                    let sample_start = area_samples.len();
                    for &(source_y, y_weight) in y_samples {
                        for &(source_x, x_weight) in x_samples {
                            let weight = y_weight * x_weight;
                            let (raw_y, raw_x) = if mask_crop {
                                (source_y, source_x)
                            } else {
                                (source_y + crop[0], source_x + crop[2])
                            };
                            let masked = mask_crop
                                && (raw_y < crop[0]
                                    || raw_y >= raw_h - crop[1]
                                    || raw_x < crop[2]
                                    || raw_x >= raw_w - crop[3]);
                            area_samples.push(AreaSample {
                                offset: if masked {
                                    MASKED_SAMPLE as u32
                                } else {
                                    (raw_y * raw_w + raw_x) as u32
                                },
                                weight: weight as u32,
                            });
                        }
                    }
                    area_pixels.push(AreaPixel {
                        sample_start: sample_start as u32,
                        sample_end: area_samples.len() as u32,
                    });
                }
            }
        }
        Self {
            raw_h,
            raw_w,
            source_h,
            source_w,
            out_h,
            out_w,
            out_c,
            crop,
            mask_crop,
            crop_fill,
            algorithm,
            nearest_y: Self::nearest_axis(source_h, out_h),
            nearest_x: Self::nearest_axis(source_w, out_w),
            linear_y: Self::linear_axis(source_h, out_h),
            linear_x: Self::linear_axis(source_w, out_w),
            area_divisor: source_h as u64 * source_w as u64,
            area_samples,
            area_pixels,
        }
    }

    fn nearest_axis(source: usize, output: usize) -> Vec<usize> {
        (0..output)
            .map(|coordinate| coordinate * source / output)
            .collect()
    }

    fn linear_axis(source: usize, output: usize) -> Vec<(usize, usize, f64)> {
        (0..output)
            .map(|coordinate| {
                let position = ((coordinate as f64 + 0.5) * source as f64 / output as f64 - 0.5)
                    .clamp(0.0, (source - 1) as f64);
                let low = position.floor() as usize;
                (low, (low + 1).min(source - 1), position - low as f64)
            })
            .collect()
    }

    fn area_axis(source: usize, output: usize) -> Vec<Vec<(usize, u64)>> {
        (0..output)
            .map(|coordinate| {
                let start = coordinate * source;
                let end = (coordinate + 1) * source;
                let first_source = start / output;
                let source_end = end.div_ceil(output).min(source);
                (first_source..source_end)
                    .map(|source_coordinate| {
                        let source_start = source_coordinate * output;
                        let source_end = (source_coordinate + 1) * output;
                        let overlap = end.min(source_end).saturating_sub(start.max(source_start));
                        (source_coordinate, overlap as u64)
                    })
                    .collect()
            })
            .collect()
    }

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
        let source_y = self.nearest_y[out_y];
        let source_x = self.nearest_x[out_x];
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
        let (y0, y1, wy) = self.linear_y[out_y];
        let (x0, x1, wx) = self.linear_x[out_x];
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
        match (self.mask_crop, previous) {
            (false, None) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                [current[offset], current[offset + 1], current[offset + 2]]
            }),
            (false, Some(prior)) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                [
                    current[offset].max(prior[offset]),
                    current[offset + 1].max(prior[offset + 1]),
                    current[offset + 2].max(prior[offset + 2]),
                ]
            }),
            (true, None) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                if offset == MASKED_SAMPLE {
                    [self.crop_fill; 3]
                } else {
                    [current[offset], current[offset + 1], current[offset + 2]]
                }
            }),
            (true, Some(prior)) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                if offset == MASKED_SAMPLE {
                    [self.crop_fill; 3]
                } else {
                    [
                        current[offset].max(prior[offset]),
                        current[offset + 1].max(prior[offset + 1]),
                        current[offset + 2].max(prior[offset + 2]),
                    ]
                }
            }),
        }
    }

    #[inline(always)]
    fn area_rgb_with<F>(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
        mut rgb_at: F,
    ) -> [u8; 3]
    where
        F: FnMut(usize) -> [u8; 3],
    {
        let pixel = &self.area_pixels[out_y * self.out_w + out_x];
        let mut sums = [0_u64; 3];
        for sample in &self.area_samples[pixel.sample_start as usize..pixel.sample_end as usize] {
            let rgb = rgb_at(if sample.offset == MASKED_SAMPLE as u32 {
                MASKED_SAMPLE
            } else {
                sample.offset as usize * 3
            });
            let weight = u64::from(sample.weight);
            sums[0] += u64::from(rgb[0]) * weight;
            sums[1] += u64::from(rgb[1]) * weight;
            sums[2] += u64::from(rgb[2]) * weight;
        }
        let integer_result = [
            ((sums[0] + self.area_divisor / 2) / self.area_divisor) as u8,
            ((sums[1] + self.area_divisor / 2) / self.area_divisor) as u8,
            ((sums[2] + self.area_divisor / 2) / self.area_divisor) as u8,
        ];
        if sums
            .iter()
            .any(|sum| (sum % self.area_divisor) * 2 == self.area_divisor)
        {
            self.area_rgb_float(current, previous, out_y, out_x)
        } else {
            integer_result
        }
    }

    #[cold]
    fn area_rgb_float(
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
    fn area_indexed_rgb(
        &self,
        current: &[u8],
        palette: &[u8],
        palette_rgb: &[u64; 256],
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let pixel = &self.area_pixels[out_y * self.out_w + out_x];
        let samples = &self.area_samples[pixel.sample_start as usize..pixel.sample_end as usize];
        let mut packed_rgb = [0_u64; 4];
        let mut chunks = samples.chunks_exact(4);
        for chunk in &mut chunks {
            for (lane, packed) in packed_rgb.iter_mut().enumerate() {
                let sample = unsafe { chunk.get_unchecked(lane) };
                let palette_index =
                    usize::from(unsafe { *current.get_unchecked(sample.offset as usize) });
                let weight = u64::from(sample.weight / INDEXED_AREA_WEIGHT_QUANTUM);
                *packed += unsafe { *palette_rgb.get_unchecked(palette_index) } * weight;
            }
        }
        for sample in chunks.remainder() {
            let palette_index =
                usize::from(unsafe { *current.get_unchecked(sample.offset as usize) });
            let weight = u64::from(sample.weight / INDEXED_AREA_WEIGHT_QUANTUM);
            packed_rgb[0] += unsafe { *palette_rgb.get_unchecked(palette_index) } * weight;
        }
        let packed_rgb = packed_rgb.into_iter().sum::<u64>();
        let sums = [
            packed_rgb & INDEXED_AREA_CHANNEL_MASK,
            (packed_rgb >> INDEXED_AREA_CHANNEL_BITS) & INDEXED_AREA_CHANNEL_MASK,
            packed_rgb >> (INDEXED_AREA_CHANNEL_BITS * 2),
        ];
        let integer_result = [
            ((sums[0] + INDEXED_AREA_DIVISOR / 2) / INDEXED_AREA_DIVISOR) as u8,
            ((sums[1] + INDEXED_AREA_DIVISOR / 2) / INDEXED_AREA_DIVISOR) as u8,
            ((sums[2] + INDEXED_AREA_DIVISOR / 2) / INDEXED_AREA_DIVISOR) as u8,
        ];
        if sums
            .iter()
            .any(|sum| (sum % INDEXED_AREA_DIVISOR) * 2 == INDEXED_AREA_DIVISOR)
        {
            self.area_indexed_rgb_float(current, palette, out_y, out_x)
        } else {
            integer_result
        }
    }

    #[cold]
    fn area_indexed_rgb_float(
        &self,
        current: &[u8],
        palette: &[u8],
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
                let raw_y = source_y + self.crop[0];
                let raw_x = source_x + self.crop[2];
                let palette_offset = usize::from(current[raw_y * self.raw_w + raw_x]) * 3;
                sums[0] += palette[palette_offset] as f64 * weight;
                sums[1] += palette[palette_offset + 1] as f64 * weight;
                sums[2] += palette[palette_offset + 2] as f64 * weight;
                weight_sum += weight;
            }
        }
        [
            (sums[0] / weight_sum).round() as u8,
            (sums[1] / weight_sum).round() as u8,
            (sums[2] / weight_sum).round() as u8,
        ]
    }

    #[inline(always)]
    fn grayscale(rgb: [u8; 3]) -> u8 {
        ((u32::from(rgb[0]) * 77 + u32::from(rgb[1]) * 150 + u32::from(rgb[2]) * 29 + 128) >> 8)
            as u8
    }

    fn write_frame(&self, current: &[u8], previous: Option<&[u8]>, output: &mut [u8]) {
        match (self.algorithm, self.out_c) {
            (ResizeAlgorithm::Nearest, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.nearest_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Bilinear, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.bilinear_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Area, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.area_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Nearest, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.nearest_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Bilinear, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.bilinear_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Area, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.area_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            _ => unreachable!("output channel count is validated at construction"),
        }
    }

    fn write_indexed_frame(&self, current: &[u8], palette: &[u8], output: &mut [u8]) {
        let mut palette_rgb = [0_u64; 256];
        for (palette_index, packed) in palette_rgb.iter_mut().enumerate() {
            let offset = palette_index * 3;
            *packed = u64::from(palette[offset])
                | (u64::from(palette[offset + 1]) << INDEXED_AREA_CHANNEL_BITS)
                | (u64::from(palette[offset + 2]) << (INDEXED_AREA_CHANNEL_BITS * 2));
        }
        for out_y in 0..self.out_h {
            for out_x in 0..self.out_w {
                output[out_y * self.out_w + out_x] = Self::grayscale(self.area_indexed_rgb(
                    current,
                    palette,
                    &palette_rgb,
                    out_y,
                    out_x,
                ));
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ObservationLayout {
    Hwc,
    Chw,
}

impl ObservationLayout {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "hwc" => Ok(Self::Hwc),
            "chw" => Ok(Self::Chw),
            _ => Err(PyValueError::new_err("layout must be 'hwc' or 'chw'")),
        }
    }
}

#[pyclass]
struct ImageProcessor {
    num_envs: usize,
    frame_stack: usize,
    layout: ObservationLayout,
    plan: ImagePlan,
    pool: ThreadPool,
}

impl ImageProcessor {
    fn validate_arrays(
        &self,
        current_shape: &[usize],
        stack_shape: &[usize],
        heads_shape: &[usize],
        output_shape: &[usize],
        previous_shape: Option<&[usize]>,
    ) -> PyResult<()> {
        let expected_current = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if current_shape != expected_current {
            return Err(PyValueError::new_err(format!(
                "current must have shape {expected_current:?}"
            )));
        }
        if stack_shape != expected_stack {
            return Err(PyValueError::new_err(format!(
                "stack must have shape {expected_stack:?}"
            )));
        }
        if heads_shape != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "heads must have shape ({},)",
                self.num_envs
            )));
        }
        if output_shape != expected_output {
            return Err(PyValueError::new_err(format!(
                "output must have shape {expected_output:?}"
            )));
        }
        if let Some(shape) = previous_shape
            && shape != expected_current
        {
            return Err(PyValueError::new_err(format!(
                "previous must have shape {expected_current:?}"
            )));
        }
        Ok(())
    }

    #[inline]
    fn write_observation(&self, stack: &[u8], head: usize, output: &mut [u8]) {
        let pixels = self.plan.out_h * self.plan.out_w;
        let frame_size = pixels * self.plan.out_c;
        match self.layout {
            ObservationLayout::Chw => {
                for output_slot in 0..self.frame_stack {
                    let source_slot = (head + 1 + output_slot) % self.frame_stack;
                    let source = &stack[source_slot * frame_size..(source_slot + 1) * frame_size];
                    if self.plan.out_c == 1 {
                        output[output_slot * pixels..(output_slot + 1) * pixels]
                            .copy_from_slice(source);
                    } else {
                        for channel in 0..self.plan.out_c {
                            let output_start = (output_slot * self.plan.out_c + channel) * pixels;
                            for pixel in 0..pixels {
                                output[output_start + pixel] =
                                    source[pixel * self.plan.out_c + channel];
                            }
                        }
                    }
                }
            }
            ObservationLayout::Hwc => {
                let stacked_channels = self.plan.out_c * self.frame_stack;
                for pixel in 0..pixels {
                    for output_slot in 0..self.frame_stack {
                        let source_slot = (head + 1 + output_slot) % self.frame_stack;
                        let source_start = source_slot * frame_size + pixel * self.plan.out_c;
                        let output_start = pixel * stacked_channels + output_slot * self.plan.out_c;
                        output[output_start..output_start + self.plan.out_c]
                            .copy_from_slice(&stack[source_start..source_start + self.plan.out_c]);
                    }
                }
            }
        }
    }
}

#[pymethods]
impl ImageProcessor {
    #[new]
    #[pyo3(signature = (
        num_envs,
        raw_height,
        raw_width,
        out_height,
        out_width,
        out_channels,
        crop,
        mask_crop,
        crop_fill,
        algorithm,
        frame_stack,
        layout,
        num_threads
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        raw_height: usize,
        raw_width: usize,
        out_height: usize,
        out_width: usize,
        out_channels: usize,
        crop: Vec<usize>,
        mask_crop: bool,
        crop_fill: u8,
        algorithm: &str,
        frame_stack: usize,
        layout: &str,
        num_threads: usize,
    ) -> PyResult<Self> {
        if num_envs == 0
            || raw_height == 0
            || raw_width == 0
            || out_height == 0
            || out_width == 0
            || frame_stack == 0
            || num_threads == 0
        {
            return Err(PyValueError::new_err(
                "dimensions, frame_stack, and num_threads must be positive",
            ));
        }
        if !matches!(out_channels, 1 | 3) {
            return Err(PyValueError::new_err("out_channels must be one or three"));
        }
        if crop.len() != 4 {
            return Err(PyValueError::new_err(
                "crop must contain top, bottom, left, right",
            ));
        }
        if crop[0] + crop[1] >= raw_height || crop[2] + crop[3] >= raw_width {
            return Err(PyValueError::new_err(
                "crop must preserve at least one source pixel",
            ));
        }
        let plan = ImagePlan::new(
            raw_height,
            raw_width,
            out_height,
            out_width,
            out_channels,
            [crop[0], crop[1], crop[2], crop[3]],
            mask_crop,
            crop_fill,
            ResizeAlgorithm::parse(algorithm)?,
        );
        let available_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.min(num_envs).min(available_threads))
            .thread_name(|index| format!("vizdoom-turbo-image-{index}"))
            .build()
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to create native image worker pool: {error}"
                ))
            })?;
        Ok(Self {
            num_envs,
            frame_stack,
            layout: ObservationLayout::parse(layout)?,
            plan,
            pool,
        })
    }

    #[pyo3(signature = (current, stack, heads, output, previous=None))]
    fn step_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray4<'_, u8>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        previous: Option<PyReadonlyArray4<'_, u8>>,
    ) -> PyResult<()> {
        self.validate_arrays(
            current.shape(),
            stack.shape(),
            heads.shape(),
            output.shape(),
            previous.as_ref().map(|array| array.shape()),
        )?;
        let current_data = current.as_slice()?;
        let previous_data = previous
            .as_ref()
            .map(|array| array.as_slice())
            .transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let raw_frame_size = self.plan.raw_h * self.plan.raw_w * 3;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(current_data.par_chunks(raw_frame_size))
                    .enumerate()
                    .for_each(
                        |(lane, (((head, stack_lane), output_lane), current_frame))| {
                            let new_head = (*head as usize + 1) % self.frame_stack;
                            let destination = &mut stack_lane
                                [new_head * image_frame_size..(new_head + 1) * image_frame_size];
                            let prior = previous_data.map(|data| {
                                &data[lane * raw_frame_size..(lane + 1) * raw_frame_size]
                            });
                            self.plan.write_frame(current_frame, prior, destination);
                            *head = new_head as i64;
                            self.write_observation(stack_lane, new_head, output_lane);
                        },
                    );
            });
        });
        Ok(())
    }

    #[pyo3(signature = (current, stack, heads, output, previous=None))]
    fn step_frames_into(
        &self,
        py: Python<'_>,
        current: Vec<PyReadonlyArray3<'_, u8>>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        previous: Option<Vec<PyReadonlyArray3<'_, u8>>>,
    ) -> PyResult<()> {
        let expected_frame = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_batch = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        self.validate_arrays(
            &expected_batch,
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if current.len() != self.num_envs
            || current.iter().any(|frame| frame.shape() != expected_frame)
        {
            return Err(PyValueError::new_err(format!(
                "current must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        if let Some(prior) = previous.as_ref()
            && (prior.len() != self.num_envs
                || prior.iter().any(|frame| frame.shape() != expected_frame))
        {
            return Err(PyValueError::new_err(format!(
                "previous must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        let current_data = current
            .iter()
            .map(|frame| frame.as_slice().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let previous_data = previous
            .as_ref()
            .map(|frames| {
                frames
                    .iter()
                    .map(|frame| frame.as_slice().map_err(PyErr::from))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .enumerate()
                    .for_each(|(lane, ((head, stack_lane), output_lane))| {
                        let new_head = (*head as usize + 1) % self.frame_stack;
                        let destination = &mut stack_lane
                            [new_head * image_frame_size..(new_head + 1) * image_frame_size];
                        let prior = previous_data.as_ref().map(|frames| frames[lane]);
                        self.plan
                            .write_frame(current_data[lane], prior, destination);
                        *head = new_head as i64;
                        self.write_observation(stack_lane, new_head, output_lane);
                    });
            });
        });
        Ok(())
    }

    fn step_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray3<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        let expected_current = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if current.shape() != expected_current {
            return Err(PyValueError::new_err(format!(
                "current must have shape {expected_current:?}"
            )));
        }
        if stack.shape() != expected_stack {
            return Err(PyValueError::new_err(format!(
                "stack must have shape {expected_stack:?}"
            )));
        }
        if head.shape() != [1] {
            return Err(PyValueError::new_err("head must have shape (1,)"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err(format!(
                "output must have shape {expected_output:?}"
            )));
        }

        let current_data = current.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        py.detach(|| {
            let new_head = (head_data[0] as usize + 1) % self.frame_stack;
            let destination =
                &mut stack_data[new_head * image_frame_size..(new_head + 1) * image_frame_size];
            self.plan.write_frame(current_data, None, destination);
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    fn step_indexed_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray2<'_, u8>,
        palette: PyReadonlyArray2<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        if self.plan.mask_crop
            || !matches!(self.plan.algorithm, ResizeAlgorithm::Area)
            || self.plan.out_c != 1
            || self.plan.raw_w != 320
            || self.plan.raw_h != 240
            || self.plan.out_w != 84
            || self.plan.out_h != 84
            || self.plan.crop != [0, 0, 0, 0]
        {
            return Err(PyValueError::new_err(
                "indexed preprocessing requires the exact 320x240 to 84x84 unmasked area-resize grayscale profile",
            ));
        }
        if current.shape() != [self.plan.raw_h, self.plan.raw_w] {
            return Err(PyValueError::new_err(
                "current has an invalid indexed shape",
            ));
        }
        if palette.shape() != [256, 3] {
            return Err(PyValueError::new_err("palette must have shape (256, 3)"));
        }
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let current_data = current.as_slice()?;
        let palette_data = palette.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        py.detach(|| {
            let new_head = (head_data[0] as usize + 1) % self.frame_stack;
            let destination =
                &mut stack_data[new_head * image_frame_size..(new_head + 1) * image_frame_size];
            self.plan
                .write_indexed_frame(current_data, palette_data, destination);
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_native_batch_into(
        &self,
        py: Python<'_>,
        context: usize,
        step_address: usize,
        frame_address: usize,
        palette_address: usize,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
    ) -> PyResult<()> {
        if self.plan.mask_crop
            || !matches!(self.plan.algorithm, ResizeAlgorithm::Area)
            || self.plan.out_c != 1
            || self.plan.raw_w != 320
            || self.plan.raw_h != 240
            || self.plan.out_w != 84
            || self.plan.out_h != 84
            || self.plan.crop != [0, 0, 0, 0]
        {
            return Err(PyValueError::new_err(
                "native batch preprocessing requires the exact 320x240 to 84x84 unmasked area-resize grayscale profile",
            ));
        }
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || heads.shape() != [self.num_envs] {
            return Err(PyValueError::new_err("stack or heads has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        type StepLane = unsafe extern "C" fn(*mut c_void, usize) -> u32;
        type BufferLane = unsafe extern "C" fn(*mut c_void, usize) -> *const u8;
        let step_lane: StepLane = unsafe { std::mem::transmute(step_address) };
        let frame_lane: BufferLane = unsafe { std::mem::transmute(frame_address) };
        let palette_lane: BufferLane = unsafe { std::mem::transmute(palette_address) };
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        let stack_lane_size = self.frame_stack * image_frame_size;
        let output_lane_size = self.frame_stack * image_frame_size;
        let failed = AtomicBool::new(false);

        py.detach(|| {
            self.pool.install(|| {
                stack_data
                    .par_chunks_mut(stack_lane_size)
                    .zip(heads_data.par_iter_mut())
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .enumerate()
                    .for_each(|(lane, ((stack_lane, head), output_lane))| {
                        let status = unsafe { step_lane(context as *mut c_void, lane) };
                        if status & 4 != 0 {
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                        let old_head = *head as usize;
                        let new_head = (old_head + 1) % self.frame_stack;
                        let destination = new_head * image_frame_size;
                        if status & 3 != 0 {
                            let source = old_head * image_frame_size;
                            if source < destination {
                                let (before, after) = stack_lane.split_at_mut(destination);
                                after[..image_frame_size]
                                    .copy_from_slice(&before[source..source + image_frame_size]);
                            } else if destination < source {
                                let (before, after) = stack_lane.split_at_mut(source);
                                before[destination..destination + image_frame_size]
                                    .copy_from_slice(&after[..image_frame_size]);
                            }
                        } else {
                            let frame = unsafe {
                                std::slice::from_raw_parts(
                                    frame_lane(context as *mut c_void, lane),
                                    self.plan.raw_h * self.plan.raw_w,
                                )
                            };
                            let palette = unsafe {
                                std::slice::from_raw_parts(
                                    palette_lane(context as *mut c_void, lane),
                                    256 * 3,
                                )
                            };
                            self.plan.write_indexed_frame(
                                frame,
                                palette,
                                &mut stack_lane[destination..destination + image_frame_size],
                            );
                        }
                        *head = new_head as i64;
                        self.write_observation(stack_lane, new_head, output_lane);
                    });
            });
        });
        if failed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("native Doom lane step failed"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reset_native_batch_into(
        &self,
        py: Python<'_>,
        context: usize,
        frame_address: usize,
        palette_address: usize,
        mask: PyReadonlyArray1<'_, bool>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
    ) -> PyResult<()> {
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if mask.shape() != [self.num_envs]
            || stack.shape() != expected_stack
            || heads.shape() != [self.num_envs]
            || output.shape() != expected_output
        {
            return Err(PyValueError::new_err(
                "native reset arrays have invalid shapes",
            ));
        }

        type BufferLane = unsafe extern "C" fn(*mut c_void, usize) -> *const u8;
        let frame_lane: BufferLane = unsafe { std::mem::transmute(frame_address) };
        let palette_lane: BufferLane = unsafe { std::mem::transmute(palette_address) };
        let mask_data = mask.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        let stack_lane_size = self.frame_stack * image_frame_size;
        let output_lane_size = self.frame_stack * image_frame_size;

        py.detach(|| {
            self.pool.install(|| {
                stack_data
                    .par_chunks_mut(stack_lane_size)
                    .zip(heads_data.par_iter_mut())
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(mask_data.par_iter())
                    .enumerate()
                    .for_each(|(lane, (((stack_lane, head), output_lane), selected))| {
                        if !*selected {
                            return;
                        }
                        let frame = unsafe {
                            std::slice::from_raw_parts(
                                frame_lane(context as *mut c_void, lane),
                                self.plan.raw_h * self.plan.raw_w,
                            )
                        };
                        let palette = unsafe {
                            std::slice::from_raw_parts(
                                palette_lane(context as *mut c_void, lane),
                                256 * 3,
                            )
                        };
                        self.plan.write_indexed_frame(
                            frame,
                            palette,
                            &mut stack_lane[..image_frame_size],
                        );
                        for slot in 1..self.frame_stack {
                            stack_lane.copy_within(..image_frame_size, slot * image_frame_size);
                        }
                        *head = 0;
                        self.write_observation(stack_lane, 0, output_lane);
                    });
            });
        });
        Ok(())
    }

    fn reset_indexed_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray2<'_, u8>,
        palette: PyReadonlyArray2<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        if self.plan.mask_crop
            || !matches!(self.plan.algorithm, ResizeAlgorithm::Area)
            || self.plan.out_c != 1
            || self.plan.raw_w != 320
            || self.plan.raw_h != 240
            || self.plan.out_w != 84
            || self.plan.out_h != 84
            || self.plan.crop != [0, 0, 0, 0]
        {
            return Err(PyValueError::new_err(
                "indexed preprocessing requires the exact 320x240 to 84x84 unmasked area-resize grayscale profile",
            ));
        }
        if current.shape() != [self.plan.raw_h, self.plan.raw_w] {
            return Err(PyValueError::new_err(
                "current has an invalid indexed shape",
            ));
        }
        if palette.shape() != [256, 3] {
            return Err(PyValueError::new_err("palette must have shape (256, 3)"));
        }
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let current_data = current.as_slice()?;
        let palette_data = palette.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        py.detach(|| {
            self.plan.write_indexed_frame(
                current_data,
                palette_data,
                &mut stack_data[..image_frame_size],
            );
            for slot in 1..self.frame_stack {
                stack_data.copy_within(..image_frame_size, slot * image_frame_size);
            }
            head_data[0] = 0;
            self.write_observation(stack_data, 0, output_data);
        });
        Ok(())
    }

    fn repeat_last_lane_into(
        &self,
        py: Python<'_>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        py.detach(|| {
            let old_head = head_data[0] as usize;
            let new_head = (old_head + 1) % self.frame_stack;
            let source = old_head * image_frame_size;
            let destination = new_head * image_frame_size;
            if source < destination {
                let (before, after) = stack_data.split_at_mut(destination);
                after[..image_frame_size]
                    .copy_from_slice(&before[source..source + image_frame_size]);
            } else if destination < source {
                let (before, after) = stack_data.split_at_mut(source);
                before[destination..destination + image_frame_size]
                    .copy_from_slice(&after[..image_frame_size]);
            }
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    fn reset_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray4<'_, u8>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        reset_mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        self.validate_arrays(
            current.shape(),
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if reset_mask.shape() != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "reset_mask must have shape ({},)",
                self.num_envs
            )));
        }
        let current_data = current.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let mask_data = reset_mask.as_slice()?;
        let raw_frame_size = self.plan.raw_h * self.plan.raw_w * 3;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(current_data.par_chunks(raw_frame_size))
                    .enumerate()
                    .for_each(
                        |(lane, (((head, stack_lane), output_lane), current_frame))| {
                            if mask_data[lane] {
                                self.plan.write_frame(
                                    current_frame,
                                    None,
                                    &mut stack_lane[..image_frame_size],
                                );
                                for slot in 1..self.frame_stack {
                                    stack_lane
                                        .copy_within(..image_frame_size, slot * image_frame_size);
                                }
                                *head = 0;
                            }
                            self.write_observation(stack_lane, *head as usize, output_lane);
                        },
                    );
            });
        });
        Ok(())
    }

    fn reset_frames_into(
        &self,
        py: Python<'_>,
        current: Vec<PyReadonlyArray3<'_, u8>>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        reset_mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        let expected_frame = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_batch = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        self.validate_arrays(
            &expected_batch,
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if current.len() != self.num_envs
            || current.iter().any(|frame| frame.shape() != expected_frame)
        {
            return Err(PyValueError::new_err(format!(
                "current must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        if reset_mask.shape() != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "reset_mask must have shape ({},)",
                self.num_envs
            )));
        }
        let current_data = current
            .iter()
            .map(|frame| frame.as_slice().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let mask_data = reset_mask.as_slice()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .enumerate()
                    .for_each(|(lane, ((head, stack_lane), output_lane))| {
                        if mask_data[lane] {
                            self.plan.write_frame(
                                current_data[lane],
                                None,
                                &mut stack_lane[..image_frame_size],
                            );
                            for slot in 1..self.frame_stack {
                                stack_lane.copy_within(..image_frame_size, slot * image_frame_size);
                            }
                            *head = 0;
                        }
                        self.write_observation(stack_lane, *head as usize, output_lane);
                    });
            });
        });
        Ok(())
    }
}

#[pyclass]
struct ActionHistory {
    action_width: usize,
    lanes: Vec<Vec<f64>>,
}

#[pymethods]
impl ActionHistory {
    #[new]
    fn new(num_envs: usize, action_width: usize) -> PyResult<Self> {
        if num_envs == 0 || action_width == 0 {
            return Err(PyValueError::new_err(
                "num_envs and action_width must be positive",
            ));
        }
        Ok(Self {
            action_width,
            lanes: vec![Vec::new(); num_envs],
        })
    }

    fn append(&mut self, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = actions.shape();
        if shape != [self.lanes.len(), self.action_width] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, {})",
                self.lanes.len(),
                self.action_width
            )));
        }
        let values = actions.as_slice()?;
        for (lane, action) in self.lanes.iter_mut().zip(values.chunks(self.action_width)) {
            lane.extend_from_slice(action);
        }
        Ok(())
    }

    fn clear(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<()> {
        if mask.shape() != [self.lanes.len()] {
            return Err(PyValueError::new_err(format!(
                "mask must have shape ({},)",
                self.lanes.len()
            )));
        }
        for (lane, &selected) in self.lanes.iter_mut().zip(mask.as_slice()?) {
            if selected {
                lane.clear();
            }
        }
        Ok(())
    }

    fn replace_lane(&mut self, lane: usize, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if lane >= self.lanes.len() {
            return Err(PyIndexError::new_err("lane is out of range"));
        }
        let shape = actions.shape();
        if shape.len() != 2 || shape[1] != self.action_width {
            return Err(PyValueError::new_err(format!(
                "actions must have shape (steps, {})",
                self.action_width
            )));
        }
        self.lanes[lane] = actions.as_slice()?.to_vec();
        Ok(())
    }

    fn lane(&self, lane: usize) -> PyResult<Vec<Vec<f64>>> {
        if lane >= self.lanes.len() {
            return Err(PyIndexError::new_err("lane is out of range"));
        }
        Ok(self.lanes[lane]
            .chunks(self.action_width)
            .map(<[f64]>::to_vec)
            .collect())
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
    let plan = ImagePlan::new(
        raw_h,
        raw_w,
        output_shape[1],
        output_shape[2],
        output_shape[3],
        [crop[0], crop[1], crop[2], crop[3]],
        mask_crop,
        crop_fill,
        ResizeAlgorithm::parse(algorithm)?,
    );
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
    module.add_class::<ActionHistory>()?;
    module.add_class::<ImageProcessor>()?;
    module.add_function(wrap_pyfunction!(preprocess_into, module)?)?;
    Ok(())
}
