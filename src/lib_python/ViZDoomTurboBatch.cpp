/*
 Copyright (C) 2026 by the ViZDoom-turbo contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "ViZDoomTurboBatch.h"
#include "ViZDoomController.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace vizdoom {

    namespace {
        template <typename T, int Flags>
        void requireShape(
            const pyb::array_t<T, Flags> &array,
            std::initializer_list<pyb::ssize_t> expected,
            const char *name) {
            if (array.ndim() != static_cast<pyb::ssize_t>(expected.size())) {
                throw std::invalid_argument(std::string(name) + " has an invalid rank");
            }
            size_t dimension = 0;
            for (pyb::ssize_t size : expected) {
                if (array.shape(dimension) != size) {
                    throw std::invalid_argument(std::string(name) + " has an invalid shape");
                }
                ++dimension;
            }
        }
    }

    TurboBatchStepper::TurboBatchStepper(
        pyb::list games,
        unsigned int frameSkip,
        bool treatTimeoutAsTruncation,
        pyb::array_t<double, pyb::array::c_style> actions,
        pyb::array_t<uint8_t, pyb::array::c_style> frames,
        pyb::array_t<uint8_t, pyb::array::c_style> palettes,
        pyb::array_t<float, pyb::array::c_style> rewards,
        pyb::array_t<bool, pyb::array::c_style> terminated,
        pyb::array_t<bool, pyb::array::c_style> truncated,
        pyb::array_t<double, pyb::array::c_style> gameVariables)
        : gameOwners(games),
          frameSkip(frameSkip),
          treatTimeoutAsTruncation(treatTimeoutAsTruncation),
          actions(std::move(actions)),
          frames(std::move(frames)),
          palettes(std::move(palettes)),
          rewards(std::move(rewards)),
          terminated(std::move(terminated)),
          truncated(std::move(truncated)),
          gameVariables(std::move(gameVariables)) {
        if (frameSkip == 0) throw std::invalid_argument("frame_skip must be positive");
        for (pyb::handle owner : games) {
            this->games.push_back(pyb::cast<DoomGamePython *>(owner));
        }
        if (this->games.empty()) throw std::invalid_argument("games must not be empty");

        const pyb::ssize_t laneCount = static_cast<pyb::ssize_t>(this->games.size());
        DoomGamePython *first = this->games.front();
        this->actionWidth = first->getAvailableButtonsSize();
        this->frameSize = first->getScreenSize();
        this->gameVariablesWidth = first->getAvailableGameVariablesSize();
        if (first->getScreenFormat() != DOOM_256_COLORS8) {
            throw std::invalid_argument("_TurboBatchStepper requires indexed screens");
        }
        for (DoomGamePython *game : this->games) {
            if (game->getAvailableButtonsSize() != this->actionWidth ||
                game->getScreenSize() != this->frameSize ||
                game->getScreenFormat() != DOOM_256_COLORS8 ||
                game->getAvailableGameVariablesSize() != this->gameVariablesWidth) {
                throw std::invalid_argument("all games must have identical Turbo layouts");
            }
        }

        requireShape(this->actions, {laneCount, static_cast<pyb::ssize_t>(this->actionWidth)}, "actions");
        requireShape(this->frames, {laneCount, first->getScreenHeight(), first->getScreenWidth()}, "frames");
        requireShape(this->palettes, {laneCount, 256, 3}, "palettes");
        requireShape(this->rewards, {laneCount}, "rewards");
        requireShape(this->terminated, {laneCount}, "terminated");
        requireShape(this->truncated, {laneCount}, "truncated");
        requireShape(
            this->gameVariables,
            {laneCount, static_cast<pyb::ssize_t>(this->gameVariablesWidth)},
            "game_variables");

        this->actionsData = this->actions.mutable_data();
        this->framesData = this->frames.mutable_data();
        this->palettesData = this->palettes.mutable_data();
        this->rewardsData = this->rewards.mutable_data();
        this->terminatedData = this->terminated.mutable_data();
        this->truncatedData = this->truncated.mutable_data();
        this->gameVariablesData = this->gameVariables.mutable_data();
    }

    void TurboBatchStepper::stepLaneInto(size_t lane) {
        if (lane >= this->games.size()) throw std::out_of_range("lane is out of range");
        ReleaseGIL gil;
        this->stepLaneNative(lane);
    }

    void TurboBatchStepper::stepLaneNative(size_t lane) {
        this->games[lane]->turboStepInto(
            this->actionsData + lane * this->actionWidth,
            this->actionWidth,
            this->frameSkip,
            nullptr,
            0,
            nullptr,
            0,
            this->rewardsData[lane],
            this->terminatedData[lane],
            this->truncatedData[lane],
            this->gameVariablesData + lane * this->gameVariablesWidth,
            this->gameVariablesWidth,
            this->treatTimeoutAsTruncation);
    }

    void TurboBatchStepper::startLaneNative(size_t lane) {
        this->games[lane]->turboStepStart(
            this->actionsData + lane * this->actionWidth,
            this->actionWidth,
            this->frameSkip);
    }

    void TurboBatchStepper::finishLaneNative(size_t lane) {
        this->games[lane]->turboStepFinish(
            nullptr,
            0,
            nullptr,
            0,
            this->rewardsData[lane],
            this->terminatedData[lane],
            this->truncatedData[lane],
            this->gameVariablesData + lane * this->gameVariablesWidth,
            this->gameVariablesWidth,
            this->treatTimeoutAsTruncation);
    }

    unsigned int TurboBatchStepper::nativeStepLane(void *context, size_t lane) noexcept {
        try {
            TurboBatchStepper *stepper = static_cast<TurboBatchStepper *>(context);
            if (lane >= stepper->games.size()) return 4;
            stepper->stepLaneNative(lane);
            return (stepper->terminatedData[lane] ? 1u : 0u) |
                (stepper->truncatedData[lane] ? 2u : 0u);
        }
        catch (...) {
            return 4;
        }
    }

    unsigned int TurboBatchStepper::nativeStartLane(void *context, size_t lane) noexcept {
        try {
            TurboBatchStepper *stepper = static_cast<TurboBatchStepper *>(context);
            if (lane >= stepper->games.size()) return 4;
            stepper->startLaneNative(lane);
            return 0;
        }
        catch (...) {
            return 4;
        }
    }

    unsigned int TurboBatchStepper::nativeFinishLane(void *context, size_t lane) noexcept {
        try {
            TurboBatchStepper *stepper = static_cast<TurboBatchStepper *>(context);
            if (lane >= stepper->games.size()) return 4;
            stepper->finishLaneNative(lane);
            return (stepper->terminatedData[lane] ? 1u : 0u) |
                (stepper->truncatedData[lane] ? 2u : 0u);
        }
        catch (...) {
            return 4;
        }
    }

    const uint8_t *TurboBatchStepper::nativeFrame(void *context, size_t lane) noexcept {
        TurboBatchStepper *stepper = static_cast<TurboBatchStepper *>(context);
        if (lane >= stepper->games.size()) return nullptr;
        return stepper->games[lane]->doomController->getScreenBuffer();
    }

    const uint8_t *TurboBatchStepper::nativePalette(void *context, size_t lane) noexcept {
        TurboBatchStepper *stepper = static_cast<TurboBatchStepper *>(context);
        if (lane >= stepper->games.size()) return nullptr;
        return stepper->games[lane]->doomController->getScreenPalette();
    }

    void TurboBatchStepper::readLaneInto(size_t lane) {
        if (lane >= this->games.size()) throw std::out_of_range("lane is out of range");
    }

    pyb::array_t<uint8_t> TurboBatchStepper::indexedFrameView(size_t lane) {
        if (lane >= this->games.size()) throw std::out_of_range("lane is out of range");
        DoomGamePython *game = this->games[lane];
        uint8_t *data = game->doomController->getScreenBuffer();
        if (data == nullptr) throw std::runtime_error("Doom screen buffer is unavailable");
        return pyb::array_t<uint8_t>(
            {game->getScreenHeight(), game->getScreenWidth()},
            {game->getScreenWidth(), 1},
            data,
            this->gameOwners[lane]);
    }

    pyb::array_t<uint8_t> TurboBatchStepper::paletteView(size_t lane) {
        if (lane >= this->games.size()) throw std::out_of_range("lane is out of range");
        DoomGamePython *game = this->games[lane];
        uint8_t *data = game->doomController->getScreenPalette();
        if (data == nullptr) throw std::runtime_error("Doom palette is unavailable");
        return pyb::array_t<uint8_t>(
            {256, 3},
            {3, 1},
            data,
            this->gameOwners[lane]);
    }

    pyb::tuple TurboBatchStepper::nativeApi() {
        return pyb::make_tuple(
            reinterpret_cast<uintptr_t>(this),
            reinterpret_cast<uintptr_t>(&TurboBatchStepper::nativeStartLane),
            reinterpret_cast<uintptr_t>(&TurboBatchStepper::nativeFinishLane),
            reinterpret_cast<uintptr_t>(&TurboBatchStepper::nativeFrame),
            reinterpret_cast<uintptr_t>(&TurboBatchStepper::nativePalette));
    }
}
