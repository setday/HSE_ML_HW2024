/***************************************************************
 * Copyright (C) 2023
 *    HSE SPb (Higher school of economics in Saint-Petersburg).
 ***************************************************************/

/* PROJECT   : ultimate_py_project
 * AUTHORS   : Serkov Alexander, Daniil Vikulov, Daniil Martsenyuk, Vasily Lebedev
 * FILE NAME : GasContainer.cxx
 * PURPOSE   : ${PURPOSE}
 *
 * No part of this file may be changed and used without agreement of
 * authors of this project.
 */

#include "GasContainer.h"


using namespace unreal_fluid::physics::gas;

GasContainer::GasContainer(int height, int width, int depth, int particle_number) : _height(height),
                                                                                    _width(width),
                                                                                    _depth(depth) {
  _storage.resize(height, std::vector<std::vector<GasCell>>(width, std::vector<GasCell>(depth)));
  _flowsX.resize(height, std::vector<std::vector<double>>(width + 1, std::vector<double>(depth)));
  _flowsY.resize(height + 1, std::vector<std::vector<double>>(width, std::vector<double>(depth)));
  _flowsZ.resize(height, std::vector<std::vector<double>>(width, std::vector<double>(depth + 1)));

  for (int counter = 0; counter < particle_number; ++counter) {
    int x = rand() % height, y = rand() % width, z = rand() % depth;

    _storage[x][y][z] = GasCell(rand() % 100, vec3f(rand() % 100, rand() % 100, rand() % 100) / 100.0);
  }

  /*for (int i = height * 4 / 6; i < height * 5 / 6; i++) {
    for (int j = width * 1 / 6; j < width * 5 / 6; j++) {
      for (int k = depth * 1 / 6; k < depth * 5 / 6; k++) {
        _storage[i][j][k] = GasCell(70, vec3f(0, 0, 1));
        _storage[i][j][k].temperature = 100;
      }
    }
  }

  for (int i = height * 1 / 6; i < height * 2 / 6; i++) {
    for (int j = width * 1 / 6; j < width * 5 / 6; j++) {
      for (int k = depth * 1 / 6; k < depth * 5 / 6; k++) {
        _storage[i][j][k] = GasCell(70, vec3f(1, 0, 0));
        _storage[i][j][k].temperature = 400;
      }
    }
  }*/
}

double GasContainer::calculateFlow(const GasCell &cell1, const GasCell &cell2, double dt) const {
  double pressureDiff = cell1.getPressure() - cell2.getPressure();
  double targetFlow = pressureDiff / 100.0;
  return targetFlow * dt;
}

void GasContainer::calculateAndSaveFlow(GasCell &cell1, GasCell &cell2, int x, int y, int z, int type, double dt) {
  double targetFlow = calculateFlow(cell1, cell2, dt);

  if (type == 0)
    _flowsY[y][x][z] = targetFlow;
  else if (type == 1)
    _flowsX[y][x][z] = targetFlow;
  else
    _flowsZ[y][x][z] = targetFlow;
}

void GasContainer::calculateFlows(double dt) {
  for (int row = 0; row < _height; ++row) {
    for (int column = 0; column < _width; ++column) {
      for (int z = 0; z < _depth; ++z) {
        auto &cell = _storage[row][column][z];

        if (row > 0)
          calculateAndSaveFlow(_storage[row - 1][column][z], cell, row, column, z, 0, dt);
        if (column > 0)
          calculateAndSaveFlow(_storage[row][column - 1][z], cell, row, column, z, 1, dt);
        if (z > 0)
          calculateAndSaveFlow(_storage[row][column][z - 1], cell, row, column, z, 2, dt);
      }
    }
  }

  for (int row = 0; row < _height; ++row) {
    for (int column = 0; column < _width; ++column) {
      for (int z = 0; z < _depth; ++z) {
        auto &cell = _storage[row][column][z];

        cell.targetFlow -= fmin(_flowsX[row][column][z], 0.0);
        cell.targetFlow += fmax(_flowsX[row][column + 1][z], 0.0);
        cell.targetFlow -= fmin(_flowsY[row][column][z], 0.0);
        cell.targetFlow += fmax(_flowsY[row + 1][column][z], 0.0);
        cell.targetFlow -= fmin(_flowsZ[row][column][z], 0.0);
        cell.targetFlow += fmax(_flowsZ[row][column][z + 1], 0.0);

        cell.realFlow = fmin(cell.targetFlow, cell.amountOfGas);
      }
    }
  }
}

double getCellFlow(const GasCell &cell, double targetFlow) {
  if (cell.targetFlow <= 0)
    return targetFlow;

  return targetFlow * cell.realFlow / cell.targetFlow;
}

void GasContainer::applyFlow(GasCell &cell1, GasCell &cell2, double targetFlow) const {
  if (targetFlow < 0)
    return applyFlow(cell2, cell1, -targetFlow);

  cell2.add(cell1.slice(getCellFlow(cell1, targetFlow)));
}

void GasContainer::applyFlows(double dt) {
  for (int row = 0; row < _height; ++row) {
    for (int column = 0; column < _width; ++column) {
      for (int z = 0; z < _depth; ++z) {
        auto &cell = _storage[row][column][z];

        if (row > 0)
          applyFlow(_storage[row - 1][column][z], cell, _flowsY[row][column][z]);
        if (column > 0)
          applyFlow(_storage[row][column - 1][z], cell, _flowsX[row][column][z]);
        if (z > 0)
          applyFlow(_storage[row][column][z - 1], cell, _flowsZ[row][column][z]);
      }
    }
  }
}

void GasContainer::diffuseTwoCells(GasCell &cell1, GasCell &cell2, double dt) {
  double slicingPart = dt * std::min(cell1.amountOfGas, cell2.amountOfGas);

  auto cell_1 = cell1.slice(slicingPart);
  auto cell_2 = cell2.slice(slicingPart);

  cell1.add(cell_2);
  cell2.add(cell_1);
}

void GasContainer::diffuseCells(double dt) {
  for (size_t row = 0; row < _height; ++row) {
    for (size_t column = 0; column < _width; ++column) {
      for (int z = 0; z < _depth; ++z) {
        auto &cell = _storage[row][column][z];
        if (row < _height - 1)
          diffuseTwoCells(cell, _storage[row + 1][column][z], dt);
        if (column < _width - 1)
          diffuseTwoCells(cell, _storage[row][column + 1][z], dt);
        if (z < _depth - 1)
          diffuseTwoCells(cell, _storage[row][column][z + 1], dt);
      }
    }
  }
}

void GasContainer::dissolveCell(GasCell &cell, double dt) {
  cell.amountOfGas -= dt * cell.amountOfGas;
  cell.amountOfGas = std::max(cell.amountOfGas, 0.0);
}

void GasContainer::dissolveCells(double dt) {
  double edgeSize = 3;

  for (int row = 0; row < _height; ++row) {
    for (int column = 0; column < _width; ++column) {
      for (int z = 0; z < _depth; ++z) {
        if (row < edgeSize || row >= _height - edgeSize ||
            column < edgeSize || column >= _width - edgeSize ||
            z < edgeSize || z >= _depth - edgeSize) {
          auto &cell = _storage[row][column][z];
          double distFromBorder = std::min(std::min(std::min(row, _height - row), std::min(column, _width - column)), std::min(z, _depth - z));
          if (distFromBorder >= edgeSize)
            continue;
          dissolveCell(cell, dt * (edgeSize - distFromBorder) / edgeSize);
        }
      }
    }
  }
}

void GasContainer::advect(double dt) {
}

unreal_fluid::physics::IPhysicalObject::Type GasContainer::getType() {
  return unreal_fluid::physics::IPhysicalObject::Type::GAS_CONTAINER_3D;
}

void *GasContainer::getData() {
  return &_storage;
}

void GasContainer::simulate(double dt) {
  calculateFlows(dt);
  applyFlows(dt);
  diffuseCells(dt / 10);
  dissolveCells(dt / 100);
}
