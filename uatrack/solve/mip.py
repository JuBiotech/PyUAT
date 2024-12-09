"""Utilizes the MIP wrapper library for solving assignment problems
"""

import numpy as np
from mip import BINARY, CBC, Model, maximize, minimize, xsum


class SimpleMIPSolver:
    """MIP based solver wrapper to solve the assignment prolbem"""

    def __init__(self, assignments, type="max", threads=1, solver_name=CBC):
        self.assignments = assignments

        sources, targets, costs, _ = assignments

        self.sources = sources
        self.targets = targets

        costs = np.concatenate(costs)

        self.num_assignments = len(costs)

        # costs are one-dimensional
        self.costs = np.array(costs).flatten()

        if solver_name == "auto":
            # let library choose installed optimizer (prefers gurobi)
            solver_name = ""

        m = Model(solver_name=solver_name)

        # set amount of threads
        m.threads = threads

        # Disable outputs
        m.verbose = 0

        self.m = m
        self.variables = []

        # create the variables for the assignment
        for _ in costs:
            self.variables.append(m.add_var(var_type=BINARY))

        self.variables = np.array(self.variables)

        if type == "max":
            opt_type = maximize
        else:
            opt_type = minimize

        # create the objective
        m.objective = opt_type(
            xsum(self.variables[i] * costs[i] for i in range(len(costs)))
        )

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
            m += xsum(self.variables[full_mask]) == 1
            # m.addConstr(gp.quicksum(self.variables[full_mask]), GRB.EQUAL, 1)

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
            m += xsum(self.variables[full_mask]) == 1
            # m.addConstr(gp.quicksum(self.variables[full_mask]), GRB.EQUAL, 1)

    def setCutOff(self, cutOff: float):
        # pylint: disable=unused-argument,no-self-use
        # self.m.Params.cutOff = cutOff
        # logging.warning("Not supported!")
        pass

    def generateSolution(self, i):

        if i != 0:
            raise NotImplementedError()

        # TODO: implement the extraction of multiple solutions

        mask = np.array([x.x >= 0.99 for x in self.variables], dtype=bool)

        # determine assignments
        used_assignments = mask  # np.array(self.assignments)[mask]

        total_costs = self.costs.dot(mask)

        return (used_assignments, total_costs)

    def generateAllSolutions(self):
        return [self.generateSolution(0)]

    def solve(self):
        """
        Search for best solution

        returns: a solution object
        """
        self.m.optimize()

        return self.generateSolution(0)

    # pylint: disable=unused-variable, unused-argument
    def populate(self, max_num_solutions, pool_gap=None):
        if max_num_solutions > 1:
            raise NotImplementedError(
                "Obtaining multiple solutions is not yet supported!"
            )

        # TODO: Implement the solution population

        # pylint: disable=pointless-string-statement
        """
        self.m.optimize()

        x: variables
        a: previous solution

        xsum(x(i) * a(i) for i in) - xsum(x(i) * (1-a(i)) for i in) <= xsum(a(i) for i in) - 1

        See: https://stackoverflow.com/questions/42591384/how-to-ask-for-second-best-solution-to-a-mip-using-jump

        """

        # self.m.optimize()
        # not yet implemented
        # logging.warning("This is not implemented")
        return [self.solve()]
        # return self.generateAllSolutions()
