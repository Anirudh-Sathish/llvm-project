//===- CustomLoopFusion.cpp - Code to perform loop fusion
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a custom affine fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <optional>
#include <sstream>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_CUSTOMAFFINELOOPFUSION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "custom-affine-loop-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {
struct LoopFusion
    : public affine::impl::CustomAffineLoopFusionBase<LoopFusion> {
  void runOnBlock(Block *block);
  void runOnOperation() override;
};
} // namespace

// fuction obtains loops that produce to a particular consumer loop
static void getProducerIds(unsigned consumerForOpId,
                           MemRefDependenceGraph *dependencyGraph,
                           SmallVectorImpl<unsigned> &producerLoopIds) {
  if (dependencyGraph->inEdges.count(consumerForOpId) == 0)
    return;
  DenseSet<Value> consumedSet;
  for (Operation *op : dependencyGraph->getNode(consumerForOpId)->loads)
    consumedSet.insert(cast<AffineReadOpInterface>(op).getMemRef());

  for (auto &edge : dependencyGraph->inEdges[consumerForOpId]) {
    auto *producerNode = dependencyGraph->getNode(edge.id);
    if (!isa<AffineForOp>(producerNode->op))
      continue;
    bool found = false;
    for (Operation *op : producerNode->stores) {
      auto storeOp = cast<AffineWriteOpInterface>(op);
      Value store = storeOp.getMemRef();
      if (consumedSet.count(store) > 0) {
        found = true;
        break;
      }
    }
    if (found == true)
      producerLoopIds.push_back(producerNode->id);
  }
}

// This function gathers memory references (memrefs) that are accessed
// by both a producer and a consumer operation.
static void
gatherProducerConsumerMemrefs(unsigned consumerId, unsigned producerId,
                              MemRefDependenceGraph *graph,
                              DenseSet<Value> &producerConsumerMemrefs) {
  auto *producerNode = graph->getNode(producerId);
  auto *consumerNode = graph->getNode(consumerId);
  DenseSet<Value> producerStores;
  for (Operation *op : producerNode->stores) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op))
      producerStores.insert(storeOp.getMemRef());
  }
  for (Operation *op : consumerNode->loads) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      if (producerStores.count(loadOp.getMemRef()) > 0) {
        producerConsumerMemrefs.insert(loadOp.getMemRef());
      }
    }
  }
}

// This function checks if a given memref is external to a specified block.
bool isExternalMemref(Value memref, Block *block) {
  auto definingOp = memref.getDefiningOp();
  if (!definingOp)
    return true;
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(definingOp))
    if (isExternalMemref(viewOp.getViewSource(), block))
      return true;

  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(definingOp, memref))
    return true;

  return llvm::any_of(memref.getUsers(), [&](Operation *user) {
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user);
    if (!ancestorOp)
      return true;
    if (ancestorOp->getBlock() != block)
      return false;
    return !isa<AffineMapAccessInterface>(*user);
  });
}

// This function extracts memory references (memrefs) that are external to
// a specific node in the MemRefDependenceGraph.
static void extractExternalMemrefs(unsigned nodeId,
                                   MemRefDependenceGraph *graph,
                                   DenseSet<Value> &externalMemrefs) {
  auto *node = graph->getNode(nodeId);
  for (Operation *op : node->stores) {
    Value memref = cast<AffineWriteOpInterface>(op).getMemRef();
    if (externalMemrefs.count(memref) > 0)
      continue;
    if (isExternalMemref(memref, &graph->block))
      externalMemrefs.insert(memref);
  }
}

// This function determines the innermost loop depth common
// to all given operations.
static unsigned getInnermostLoopDepth(ArrayRef<Operation *> ops) {
  unsigned numOps = ops.size();

  std::vector<SmallVector<AffineForOp, 4>> loops(numOps);
  unsigned loopDepthLimit = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < numOps; ++i) {
    getAffineForIVs(*ops[i], &loops[i]);
    loopDepthLimit =
        std::min(loopDepthLimit, static_cast<unsigned>(loops[i].size()));
  }

  unsigned loopDepth = 0;
  for (unsigned d = 0; d < loopDepthLimit; ++d) {
    unsigned i;
    for (i = 1; i < numOps; ++i) {
      if (loops[i - 1][d] != loops[i][d])
        return loopDepth;
    }
    ++loopDepth;
  }
  return loopDepth;
}

// This function gathers all load
// and store operations within a given affine for loop
static bool getMemrefs(AffineForOp forOp, SmallVectorImpl<Operation *> &ops) {
  bool hasIfOp = false;
  forOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      ops.push_back(op);
    else if (isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

// This function returns the number of common surrounding loops
// shared by two operations.
static unsigned getCommonSourroundingLoops(Operation &firstOp,
                                           Operation &secondOp) {
  SmallVector<Value, 4> loop1, loop2;
  getAffineIVs(firstOp, loop1);
  getAffineIVs(secondOp, loop2);
  unsigned minNumLoops = std::min(loop1.size(), loop2.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loop1[i] != loop2[i])
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

// This function fuses a producer affine for loop into a
// consumer affine for loop based on an optimal slice state.
static void fuse(AffineForOp producerAffineForOp, AffineForOp consumerForOp,
                 ComputationSliceState &optSlice) {
  OpBuilder builder(optSlice.insertPoint->getBlock(), optSlice.insertPoint);
  IRMapping mapper;
  builder.clone(*producerAffineForOp, mapper);

  SmallVector<AffineForOp, 8> slices;
  for (auto i = 0; i < optSlice.ivs.size(); ++i) {
    auto iv = mapper.lookupOrNull(optSlice.ivs[i]);
    if (!iv)
      continue;
    auto forOp = getForInductionVarOwner(iv);
    slices.push_back(forOp);
    if (AffineMap lbMap = optSlice.lbs[i]) {
      auto lbOperands = optSlice.lbOperands[i];
      canonicalizeMapAndOperands(&lbMap, &lbOperands);
      forOp.setLowerBound(lbOperands, lbMap);
    }
    if (AffineMap ubMap = optSlice.ubs[i]) {
      auto ubOperands = optSlice.ubOperands[i];
      canonicalizeMapAndOperands(&ubMap, &ubOperands);
      forOp.setUpperBound(ubOperands, ubMap);
    }
  }
}

// This function checks the feasibility of fusing a producer
// affine for loop into a consumer affine for loop.
static bool fuseFeasibility(AffineForOp producerAffineForOp,
                            AffineForOp consumerForOp,
                            unsigned consumerLoopDepth,
                            ComputationSliceState *producerSlice) {
  if (consumerLoopDepth == 0)
    return false;
  auto *producerBlock = producerAffineForOp->getBlock();
  if (producerBlock != consumerForOp->getBlock())
    return false;

  bool correctlyOrdered = producerAffineForOp->isBeforeInBlock(consumerForOp);
  auto firstFor = correctlyOrdered ? producerAffineForOp : consumerForOp;
  auto secondFor = correctlyOrdered ? consumerForOp : producerAffineForOp;

  // gather all loads and stores from firstFOr that precede the second
  // for in the block
  SmallVector<Operation *, 4> firstOpMemrefs;
  if (!(getMemrefs(firstFor, firstOpMemrefs)))
    return false;

  // gathering loads and stores gt that in the block that succeed the first
  // loop
  SmallVector<Operation *, 4> secondOpMemrefs;
  if (!(getMemrefs(secondFor, secondOpMemrefs)))
    return false;

  unsigned commonLoopsCount =
      getCommonSourroundingLoops(*producerAffineForOp, *consumerForOp);

  // filter out ops in firstOp memrefs to form slice using
  // producer consumer
  SmallVector<Operation *, 4> firstLoopStores;
  for (Operation *op : firstOpMemrefs) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op))
      firstLoopStores.push_back(storeOp);
  }
  SliceComputationResult resultSlice = affine::computeSliceUnion(
      firstLoopStores, secondOpMemrefs, consumerLoopDepth, commonLoopsCount,
      correctlyOrdered, producerSlice);
  if (resultSlice.value == SliceComputationResult::GenericFailure)
    return false;
  if (resultSlice.value == SliceComputationResult::IncorrectSliceFailure)
    return false;

  return true;
}


namespace {
struct FusionPolicy {
  MemRefDependenceGraph *dependencyGraph;
  SmallVector<unsigned, 8> workList;

  using Node = MemRefDependenceGraph::Node;
  // Constructor to initialize the FusionPolicy with the given dependency graph
  FusionPolicy(MemRefDependenceGraph *dependencyGraph)
      : dependencyGraph(dependencyGraph) {}

   // Initialize the worklist with all nodes in the dependency graph
  void init() {
    workList.clear();
    for (auto &nodeContainer : dependencyGraph->nodes) {
      const Node &node = nodeContainer.second;
      workList.push_back(node.id);
    }
  }
  // Function to initiate the producer-consumer fusion process
  void runProducerConsumerFusion() { fuseNodes(); }

  // Main function to handle node fusion by processing the worklist
  void fuseNodes() {
    init();
    while (!workList.empty()) {
      unsigned consumerForOpId = workList.back();
      workList.pop_back();
      fuseNodesIntoConsumer(consumerForOpId);
    }
  }
  // Attempt to fuse nodes into the given consumer node
  void fuseNodesIntoConsumer(unsigned consumerForOpId) {
    if (dependencyGraph->nodes.count(consumerForOpId) == 0)
      return;
    Node *consumerNode = dependencyGraph->getNode(consumerForOpId);
    if (!isa<AffineForOp>(consumerNode->op))
      return;
    AffineForOp consumerForOp = cast<AffineForOp>(consumerNode->op);

    bool nodeUpdated = true;
    while (nodeUpdated) {
      nodeUpdated = false;
      SmallVector<unsigned, 8> producerLoopIds;
      getProducerIds(consumerForOpId, dependencyGraph, producerLoopIds);
      for (unsigned producerId : llvm::reverse(producerLoopIds)) {
        auto *producerNode = dependencyGraph->getNode(producerId);
        auto producerAffineForOp = cast<AffineForOp>(producerNode->op);
        if (isa<AffineForOp>(producerNode->op) &&
            producerNode->op->getNumResults() > 0)
          continue;

        DenseSet<Value> producerConsumerMemrefs;
        gatherProducerConsumerMemrefs(consumerForOpId, producerId,
                                      dependencyGraph, producerConsumerMemrefs);
        
        DenseSet<Value> externalMemrefs;
        extractExternalMemrefs(producerId, dependencyGraph, externalMemrefs);

        Operation *insertionPoint =
            dependencyGraph->getFusedLoopNestInsertionPoint(producerNode->id,
                                                            consumerNode->id);
        if (insertionPoint == nullptr)
          continue;
        
        SmallVector<AffineForOp, 4> surroundingLoops;
        getAffineForIVs(*consumerForOp, &surroundingLoops);
        unsigned numSurroundingLoops = surroundingLoops.size();

        SmallVector<Operation *, 2> consumerMemrefOps;
        for (Operation *op : consumerNode->loads)
          if (producerConsumerMemrefs.count(
                  cast<AffineReadOpInterface>(op).getMemRef()) > 0)
            consumerMemrefOps.push_back(op);
        for (Operation *op : consumerNode->stores)
          if (producerConsumerMemrefs.count(
                  cast<AffineWriteOpInterface>(op).getMemRef()))
            consumerMemrefOps.push_back(op);
        unsigned consumerLoopDepthTest =
            getInnermostLoopDepth(consumerMemrefOps) - numSurroundingLoops;
        SmallVector<ComputationSliceState, 8> loopSlices;
        loopSlices.resize(consumerLoopDepthTest);
        unsigned fusionDepth = 0;
        for (unsigned i = 1; i <= consumerLoopDepthTest; ++i) {
          bool result = fuseFeasibility(producerAffineForOp, consumerForOp,
                                        i + numSurroundingLoops,
                                        &loopSlices[i - 1]);
          if (result)
            fusionDepth = i;
        }
        if (fusionDepth == 0)
          return;
        ComputationSliceState &optSlice = loopSlices[fusionDepth - 1];
        if (optSlice.isEmpty())
          return;
        fuseLoops(producerAffineForOp, consumerForOp, optSlice);
        nodeUpdated = true;
        if (insertionPoint != consumerForOp)
          consumerForOp->moveBefore(insertionPoint);
        LoopNestStateCollector consumerLoopCollector;
        consumerLoopCollector.collect(consumerForOp);
        dependencyGraph->clearNodeLoadAndStores(consumerNode->id);
        dependencyGraph->addToNode(consumerNode->id,
                                   consumerLoopCollector.loadOpInsts,
                                   consumerLoopCollector.storeOpInsts);
        producerAffineForOp.erase();
        dependencyGraph->removeNode(producerId);
        producerNode = nullptr;
      }
    }
  }
};
} // namespace

// runs loop fusion on the block
void LoopFusion::runOnBlock(Block *block) {
  MemRefDependenceGraph graph(*block);
  if (!graph.init()) {
    return;
  }
  FusionPolicy fusion(&graph);
  fusion.runProducerConsumerFusion();
}

// Function to run loop fusion on the operation
void LoopFusion::runOnOperation() {
  Operation *parentOp = getOperation();
  parentOp->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors)) {
          runOnBlock(&block);
        }
      }
    }
  });
}

// Registering the custom loop fusion pass
std::unique_ptr<Pass> mlir::affine::createCustomLoopFusion() {
  return std::make_unique<LoopFusion>();
}