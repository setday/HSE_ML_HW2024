/***************************************************************
 * Copyright (C) 2023
 *    HSE SPb (Higher school of economics in Saint-Petersburg).
 ***************************************************************/

/* PROJECT   : UnrealFluidPhysics
 * AUTHORS   : Serkov Alexander, Daniil Vikulov, Daniil Martsenyuk, Vasily Lebedev
 * FILE NAME : SimpleFluidContainer.cxx
 * PURPOSE   : Simple fluid simulation class, which implements ASS collision
 *
 * No part of this file may be changed and used without agreement of
 * authors of this project.
 */

#include "SimpleFluidContainer.h"
#include <random>
#include "../../CollisionSolver.h"

using namespace unreal_fluid::physics::fluid;

SimpleFluidContainer::SimpleFluidContainer(FluidDescriptor descriptor) : distributor(CellsDistributor()) {
  k = 0.1;
  /// TODO : write constructor implementation
}

SimpleFluidContainer::~SimpleFluidContainer() {
  for (auto &particle: particles) {
    delete particle;
  }
}

void SimpleFluidContainer::addExternalForces(double dt) {
  for (auto &particle: particles) {
    particle->velocity += G * dt;
  }
}

void SimpleFluidContainer::advect(double dt) {
  for (auto particle: particles)
    particle->position += particle->velocity * dt;

  /// TODO this is the temporary measure to prevent particles from falling down. Solids should be used
  for (auto p: particles) {
    double push = -1 - p->position.y + p->radius;
    if (push > 0) {
      p->position.y += push;
      p->velocity.y = -k * p->velocity.y;
    }
  }
}

void SimpleFluidContainer::interact() {
  distributor.update(particles);

  for (auto bigParticle: distributor.big_particles) {
    for (auto particle: particles) {
      if (particle->position != bigParticle->position &&
          (particle->position - bigParticle->position).len() <= particle->radius + bigParticle->radius)
        CollisionSolver::particleWithParticleCollision(particle, bigParticle, k);
    }
  }

  for (auto p = distributor.nextPair(); p != CellsDistributor::terminator; p = distributor.nextPair())
    CollisionSolver::particleWithParticleCollision(p.first, p.second, k);
}

/// TODO flows and addParticle should be methods of another class
void SimpleFluidContainer::flows() {
  for (int i = 0; i < 10; ++i) {
    addParticle({double(rand() % 100) / 100000, 1, double(rand() % 100) / 100000}, {0, 0, 0}, 0.02, 1);
  }
}

void SimpleFluidContainer::simulate(double dt) {
  flows();
  interact();
  addExternalForces(dt);
  advect(dt);
}

void *SimpleFluidContainer::getData() {
  return &particles;
}

unreal_fluid::physics::IPhysicalObject::Type SimpleFluidContainer::getType() {
  return Type::SIMPLE_FLUID_CONTAINER;
}

void SimpleFluidContainer::addParticle(vec3 position, vec3 velocity, double radius, double mass) {
  auto particle = new Particle;
  particle->position = position;
  particle->mass = mass;
  particle->radius = radius;
  particle->velocity = velocity;
  particles.push_back(particle);
}

// end of FluidContainer.cxx
