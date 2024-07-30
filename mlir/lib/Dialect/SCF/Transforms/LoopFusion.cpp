//===- ForToWhile.cpp - scf.for to scf.while loop conversion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Performs producer conumser fusion on SCF ForOp
// Currently supports 1D , 2D and reduction types
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <optional>
#include <sstream>
namespace mlir {
#define GEN_PASS_DEF_SCFLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir
using namespace llvm;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::affine;
using memref::LoadOp;
using memref::StoreOp;
using scf::ForOp;
#define DEBUG_TYPE "scf-loop-fusion"

// Sturct for the dependence graph to represent memref relationships
// Currently support for producer consumer dependence(RAW)
struct DependenceGraph {
  struct Edge {
    unsigned dstId;
    Value value;
    Edge(unsigned dstId, Value value) : dstId(dstId), value(value) {}
  };
  struct Node {
    unsigned id;
    SmallVector<Operation *> storeInsts;
    SmallVector<Operation *> loadInsts;
    SmallVector<Edge> edgeList;
    Operation *op;
    Node() : id(0), op(nullptr) {}
    Node(unsigned id, Operation *op) : id(id), op(op) {};
  };
  llvm::DenseMap<unsigned, std::unique_ptr<Node>> nodeMap;
  unsigned nodeId = 0;
  Block &block;
  SmallVector<unsigned> nodeList;
  DependenceGraph(Block &block) : block(block) {}

  // Walks through the given operation(which is of type ForOp) and
  // collects all the stores and loads related to it
  // If there is a nested forOp, the function is visited recursively
  void walk_operation(Operation *op, unsigned id) {
    LLVM_DEBUG(llvm::dbgs() << "============================= \n");
    op->walk([&](Operation *childOp) {
      if (isa<ForOp>(childOp)) {
        if (childOp != op) {
          nodeId++;
          auto node = std::make_unique<Node>(nodeId, childOp);
          nodeMap[nodeId] = std::move(node);
          nodeList.push_back(nodeId);
          walk_operation(childOp, nodeId);
        }
      }
      if (isa<StoreOp>(childOp)) {
        nodeMap[id]->storeInsts.push_back(childOp);
      }
      if (isa<LoadOp>(childOp)) {
        nodeMap[id]->loadInsts.push_back(childOp);
      }
    });
  }

  // Initalises the graph
  void init() {
    nodeId = 0;
    nodeMap.clear();
    nodeList.clear();
    for (Operation &op : block) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        nodeId++;
        auto node = std::make_unique<Node>(nodeId, &op);
        nodeMap[nodeId] = std::move(node);
        nodeList.push_back(nodeId);
        walk_operation(&op, nodeId);
      }
    }

    // for edge creation
    for (unsigned srcId : nodeList) {
      for (auto store : nodeMap[srcId]->storeInsts) {
        auto storeOp = cast<StoreOp>(store);
        Value memrefStore = storeOp.getMemRef();
        for (unsigned dstId : nodeList) {
          if (srcId == dstId)
            continue;
          for (auto load : nodeMap[dstId]->loadInsts) {
            auto loadOp = cast<LoadOp>(load);
            Value memrefLoad = loadOp.getMemRef();
            if (storeOp.getMemRef() == loadOp.getMemRef()) {
              nodeMap[srcId]->edgeList.push_back(Edge(dstId, memrefLoad));
            }
          }
        }
      }
    }

    for (unsigned id : nodeList) {
      if (nodeMap[id]->edgeList.size()) {
        std::sort(
            nodeMap[id]->edgeList.begin(), nodeMap[id]->edgeList.end(),
            [](const Edge &a, const Edge &b) { return a.dstId > b.dstId; });
      }
    }
  }
};

// Checks if it is feasible to fuse two scf for ops
bool checkFeasibilty(ForOp srcForOp, ForOp dstForOp) {
  auto srcLb = srcForOp.getLowerBound();
  auto srcUb = srcForOp.getUpperBound();
  auto srcStep = srcForOp.getStep();

  // Get the upper bound, lower bound, and step for the destination loop
  auto dstLb = dstForOp.getLowerBound();
  auto dstUb = dstForOp.getUpperBound();
  auto dstStep = dstForOp.getStep();
  if (srcLb == dstLb && srcUb == dstUb && srcStep == dstStep) {
    llvm::outs() << "The loops have the same bounds and step.\n";
    return true;
  }
  return false;
}

// Obtains the loop depth of a particular opearation
unsigned getLoopDepth(mlir::Operation *op) {
  unsigned depth = 0;
  while (op) {
    if (llvm::isa<mlir::scf::ForOp>(op)) {
      depth++;
    }
    op = op->getParentOp();
  }
  return depth;
}

// Obtains the maximum loop depth among all the operations
// in the initalised block for Dependence Graph
unsigned getMaxLoopDepth(DependenceGraph &graph) {
  unsigned maxDepth = 1, depth = 1;
  for (unsigned id : llvm::reverse(graph.nodeList)) {
    Operation *defOp = graph.nodeMap[id]->op;
    if (auto forOp = dyn_cast<ForOp>(defOp)) {
      depth = getLoopDepth(forOp.getOperation());
      if (depth > maxDepth)
        maxDepth = depth;
    }
  }
  return maxDepth;
}

// Fuses a given source ForOp to a destination ForOp
void fuseIntoDst(unsigned srcId, unsigned dstId, DependenceGraph &graph,
                 unsigned depth) {
  if (graph.nodeMap.find(srcId) == graph.nodeMap.end() ||
      graph.nodeMap.find(dstId) == graph.nodeMap.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid node IDs: " << srcId << " or " << dstId << "\n");
    return;
  }
  ForOp srcForOp = dyn_cast<ForOp>(graph.nodeMap[srcId]->op);
  ForOp dstForOp = dyn_cast<ForOp>(graph.nodeMap[dstId]->op);
  if (!srcForOp || !dstForOp) {
    llvm::errs() << "One of the operations is not an ForOp.\n";
    return;
  }
  if (!checkFeasibilty)
    return;
  unsigned srcLoopDepth = getLoopDepth(srcForOp.getOperation());
  unsigned dstLoopDepth = getLoopDepth(dstForOp.getOperation());
  if (srcLoopDepth != dstLoopDepth)
    return;
  if (dstLoopDepth != depth)
    return;
  Value dstIV = dstForOp.getInductionVar();
  Value srcIV = srcForOp.getInductionVar();
  IRMapping mapper;
  mapper.map(srcIV, dstIV);
  Block *dstBody = dstForOp.getBody();
  auto firstOp = dstBody->begin();
  OpBuilder builder(dstBody, firstOp);
  for (Operation &op : srcForOp.getBody()->without_terminator()) {
    builder.clone(op, mapper);
  }
  srcForOp.erase();
}

// Fuses the loops in a given block , initalised in the Dependence Graph
void fuseLoops(DependenceGraph &graph) {
  unsigned maxDepth = getMaxLoopDepth(graph);
  for (unsigned depth = 1; depth <= maxDepth; depth++) {
    for (unsigned id : llvm::reverse(graph.nodeList)) {
      if (graph.nodeMap[id]->edgeList.size()) {
        for (auto edge : graph.nodeMap[id]->edgeList) {
          if (graph.nodeMap.find(edge.dstId) == graph.nodeMap.end())
            continue;
          Operation *srcOp = graph.nodeMap[id]->op;
          Operation *dstOp = graph.nodeMap[edge.dstId]->op;

          // Check if the operations are valid
          if (!srcOp || !dstOp) {
            LLVM_DEBUG(llvm::dbgs() << "Invalid operations for srcId: " << id
                                    << " or dstId: " << edge.dstId << "\n");
            continue;
          }

          LLVM_DEBUG(llvm::dbgs() << "src op: " << *srcOp << "\n");
          LLVM_DEBUG(llvm::dbgs() << "dst op: " << *dstOp << "\n");

          fuseIntoDst(id, edge.dstId, graph, depth);
        }
      }
    }
    graph.init();
  }
}

namespace {
struct LoopFusion : public impl::SCFLoopFusionBase<LoopFusion> {
  void runOnBlock(Block *block);
  void runOnOperation() override;
};
} // namespace

// Performs loop fusion on a particular block
void LoopFusion::runOnBlock(Block *block) {
  DependenceGraph dg(*block);
  dg.init();
  fuseLoops(dg);
}

// Obtains blocks where loop fusion can be applied and perform loop fusion
void LoopFusion::runOnOperation() {
  Operation *parentOp = getOperation();
  parentOp->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto fors = block.getOps<ForOp>();
        if (!fors.empty() && !llvm::hasSingleElement(fors)) {
          runOnBlock(&block);
        }
      }
    }
  });
}

std::unique_ptr<Pass> mlir::createLoopFusion() {
  return std::make_unique<LoopFusion>();
}