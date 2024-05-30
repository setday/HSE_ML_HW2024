/***************************************************************
 * Copyright (C) 2023
 *    HSE SPb (Higher school of economics in Saint-Petersburg).
 ***************************************************************/

/* PROJECT   : ultimate_py_project
 * AUTHORS   : Serkov Alexander, Daniil Vikulov, Daniil Martsenyuk, Vasily Lebedev
 * FILE NAME : SceneCompositor.cxx
 * PURPOSE   : This class is responsible for rendering.
 *
 * No part of this file may be changed and used without agreement of
 * authors of this project.
 */

#include "SceneCompositor.h"
#include "./../../scenes/SceneLoader.cxx"
#include "components/scene/Scene.h"

using namespace unreal_fluid::compositor;

SceneCompositor::SceneCompositor(Core *core) : _core(core) {
  Logger::logInfo("Creating compositor...");

  _timer.pause();
  _timer.reset();
  _renderingTimer.pause();
  _renderingTimer.reset();
  _simulationTimer.pause();
  _simulationTimer.reset();

  Logger::logInfo("SceneCompositor created!");
}

void SceneCompositor::init() {
  Logger::logInfo("Initializing compositor...");

  _simulator = new physics::Simulator;
  _renderer = std::make_unique<render::Renderer>();

  loadScene<SceneLoader>();

  Logger::logInfo("SceneCompositor initialized!");
}

void SceneCompositor::update() {
  _timer.resume();
  _simulationTimer.resume();

  for (auto scene: _scenes) {
    if (scene != nullptr)
      scene->update();
  }

  _simulationTimer.pause();
  _timer.pause();
  _simulationTimer.incrementCounter();
}

void SceneCompositor::render() {
  _timer.resume();
  _renderingTimer.resume();

  _renderer->startFrame();
  for (auto scene: _scenes) {
    if (scene != nullptr)
      scene->render();
  }
  _renderer->endFrame();

  _renderingTimer.pause();
  _timer.pause();
  _renderingTimer.incrementCounter();
  _timer.incrementCounter();

  if (_timer.getCounter() >= 80 || _timer.getElapsedTime() >= 1.5) {
    Logger::logInfo(
            "\n", "-------------------------------------------->\n",
            "| Time to crate a frame: ", _timer.getAverageTime<utils::Timer::TimeType::MILLISECONDS>(), " ms\n",
            "| FPS: ", 1 / _timer.getAverageTime(), "\n",
            "-------------------------------------------->\n",
            "| Time to render a frame: ", _renderingTimer.getAverageTime<utils::Timer::TimeType::MILLISECONDS>(), " ms\n",
            "| Time to simulate a frame: ", _simulationTimer.getAverageTime<utils::Timer::TimeType::MILLISECONDS>(), " ms\n",
            "-------------------------------------------->"
    );
    _timer.reset();
    _renderingTimer.reset();
    _simulationTimer.reset();
  }
}

void SceneCompositor::destroy() {
  for (auto scene: _scenes)
    delete scene;

  _renderer.reset();
}

void SceneCompositor::unloadScene(IScene *scene) {
  for (auto it = _scenes.begin(); it != _scenes.end(); ++it) {
    if (*it == scene) {
      _scenes.erase(it);
      delete scene;
      break;
    }
  }
}

Core *SceneCompositor::getCore() const {
  return _core;
}

physics::Simulator *SceneCompositor::getSimulator() const {
  return _simulator;
}

render::Renderer *SceneCompositor::getRenderer() const {
  return _renderer.get();
}

// end of SceneCompositor.cxx
