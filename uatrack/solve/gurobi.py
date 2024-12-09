"""Utilize the gurobi interface for solving the assignment problem"""

import logging

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class SimpleGurobiSolver:
    """Gurobi solver wrapper for the assignment problem"""

    def __init__(self, assignments, type="max", threads=1):
        self.assignments = assignments

        sources, targets, costs, _ = assignments

        self.sources = sources
        self.targets = targets

        costs = np.concatenate(costs)

        self.num_assignments = len(costs)

        # costs are one-dimensional
        costs = np.array(costs).flatten()

        gp.setParam(GRB.Param.Threads, threads)
        gp.setParam(GRB.Param.OutputFlag, 0)

        m = gp.Model("expansion")

        self.m = m
        self.variables = []

        # create the variables for the assignment
        for _ in costs:
            self.variables.append(m.addVar(vtype=GRB.BINARY))

        self.variables = np.array(self.variables)

        if type == "max":
            grb_type = GRB.MAXIMIZE
        else:
            grb_type = GRB.MINIMIZE

        # create the objective
        expr = gp.LinExpr()
        for i, _ in enumerate(costs):
            expr.add(self.variables[i], costs[i])
        m.setObjective(expr, grb_type)

        all_sources = np.concatenate([s.flatten() for s in sources])
        valid_sources = all_sources[all_sources >= 0]
        unqiue_sources = np.unique(valid_sources)
        unique_targets = np.unique(np.concatenate([t.flatten() for t in targets]))

        # source touched exactly once
        for source in unqiue_sources:
            all_masks = []
            for ass_sources in sources:
                if ass_sources.shape[1] == 0:
                    # in case we have no sources
                    all_masks.append(np.zeros(ass_sources.shape[0], dtype=bool))
                    continue

                assignment_mask = np.isin(ass_sources, source)
                if len(assignment_mask.shape) == 2:
                    assignment_mask = np.any(assignment_mask, axis=-1)
                all_masks.append(assignment_mask)

            full_mask = np.concatenate(all_masks)

            # add a joint constraint for all the assignments that are masked true
            m.addConstr(gp.quicksum(self.variables[full_mask]), GRB.EQUAL, 1)

        # target touched exactly once
        for target in unique_targets:
            all_masks = []
            for ass_targets in targets:
                if ass_targets.shape[1] == 0:
                    # in case we have no targets
                    all_masks.append(np.zeros(ass_targets.shape[0], dtype=bool))
                    continue

                assignment_mask = np.isin(ass_targets, target)
                if len(assignment_mask.shape) == 2:
                    assignment_mask = np.any(assignment_mask, axis=-1)
                all_masks.append(assignment_mask)

            full_mask = np.concatenate(all_masks)

            # add a joint constraint for all the assignments that are masked true
            m.addConstr(gp.quicksum(self.variables[full_mask]), GRB.EQUAL, 1)

    def setCutOff(self, cutOff: float):
        self.m.Params.cutOff = cutOff

    def generateSolution(self, i):
        # check whether enough solutions are available
        if i >= self.m.SolCount:
            raise ValueError(
                f"You want to get solution {i} but only {self.m.SolCount} are available!"
            )

        # select the corresponding solution
        self.m.setParam(GRB.Param.SolutionNumber, i)

        # determine assignments
        used_assignments = [
            self.variables[i].Xn > 0.9 for i, _ in enumerate(self.variables)
        ]

        total_costs = self.m.PoolObjVal

        return (used_assignments, total_costs)

    def generateAllSolutions(self):
        return [self.generateSolution(i) for i in range(self.m.SolCount)]

    def solve(self):
        """
        Search for best solution

        returns: a solution object
        """
        self.m.setParam(GRB.Param.PoolSearchMode, 0)

        self.m.optimize()

        return self.generateSolution(0)

    def populate(self, max_num_solutions, pool_gap=None):
        # Limit how many solutions to collect
        self.m.setParam(GRB.Param.PoolSolutions, max_num_solutions)
        # Limit the search space by setting a gap for the worst possible solution
        # that will be accepted
        if pool_gap:
            self.m.setParam(GRB.Param.PoolGap, pool_gap)
        # do a systematic search for the k-best solutions
        self.m.setParam(GRB.Param.PoolSearchMode, 2)

        # Optimize model
        self.m.optimize()

        nSolutions = self.m.SolCount

        logging.info(f"Number of solutions: {nSolutions}")

        if self.m.Status == GRB.TIME_LIMIT:
            logging.warning("Time limit reached! Solutions found %d", nSolutions)

        try:
            return [self.generateSolution(i) for i in range(nSolutions)]
        except AttributeError:
            logging.warning(
                "Attribute error while generating solutions! No solutions found!"
            )
            return []
