#constraints
from pacman.model.constraints.abstract_partitioner_constraint import \
    AbstractPartitionerConstraint
from pacman.model.constraints.partitioner_maximum_size_constraint import \
    PartitionerMaximumSizeConstraint
from pacman.model.constraints.partitioner_same_size_as_vertex_constraint \
    import PartitionerSameSizeAsVertexConstraint
from pacman.model.constraints.placer_chip_and_core_constraint\
    import PlacerChipAndCoreConstraint
from pacman.model.constraints.placer_subvertex_same_chip_constraint import \
    PlacerSubvertexSameChipConstraint

#objects
from pacman.model.graph_mapper.graph_mapper import GraphMapper
from pacman.model.partitioned_graph.partitioned_graph import PartitionedGraph
from pacman.model.graph_mapper.slice import Slice

#algorithums
from pacman.operations.partition_algorithms.abstract_partition_algorithm \
    import AbstractPartitionAlgorithm
from pacman.operations.placer_algorithms.abstract_placer_algorithm import \
    AbstractPlacerAlgorithm

#extras
from pacman import exceptions
from pacman.utilities import utility_calls

import math
import logging
import sys

logger = logging.getLogger(__name__)


class SpreaderPartitioner(AbstractPartitionAlgorithm):
    """ An basic algorithm that can partition a partitionable_graph based
    on atoms
    """

    def __init__(self, machine_time_step, runtime_in_machine_time_steps):
        """constructor to build a
    pacman.operations.partition_algorithms.basic_partitioner.BasicPartitioner

        :param machine_time_step: the length of tiem in ms for a timer tick
        :param runtime_in_machine_time_steps: the number of timer ticks
        expected to occur due to the runtime
        :type machine_time_step: int
        :type runtime_in_machine_time_steps: long
        :return: a new
        pacman.operations.partition_algorithms.
        abstract_partition_algorithm.AbstractPartitionAlgorithm
        :rtype: pacman.operations.partition_algorithms.
        abstract_partition_algorithm.AbstractPartitionAlgorithm
        :raises None: does not raise any known expection
        """
        AbstractPartitionAlgorithm.__init__(self, machine_time_step,
                                            runtime_in_machine_time_steps)

        #add supported constraints
        self._supported_constraints.append(PartitionerMaximumSizeConstraint)
        self._supported_constraints.\
            append(PartitionerSameSizeAsVertexConstraint)
        self._supported_constraints.append(PlacerChipAndCoreConstraint)
        self._supported_constraints.append(PlacerSubvertexSameChipConstraint)
        # placer algorithm for figureing resources of machine
        self._placer_algorithm = None

    #inherited from AbstractPartitionAlgorithm
    def partition(self, graph, machine):
        """ Partition a partitionable_graph so that each
        subvertex will fit on a processor within the machine

        :param graph: The partitionable_graph to partition
        :type graph: :py:class:`pacman.model.graph.partitionable_graph.
        PartitionableGraph`
        :param machine: The machine with respect to which to partition the
        partitionable_graph
        :type machine: :py:class:`spinn_machine.machine.Machine`
        :return: A partitioned_graph of partitioned vertices and edges from
        the partitionable_graph
        :rtype: :py:class:`pacman.model.subgraph.subgraph.Subgraph`
        :raise pacman.exceptions.PacmanPartitionException: If something\
                   goes wrong with the partitioning
        """

        utility_calls.check_algorithm_can_support_constraints(
            constrained_vertices=graph.vertices,
            supported_constraints=self._supported_constraints,
            abstract_constraint_type=AbstractPartitionerConstraint)

        logger.info("* Running Spread Partitioner *")

        # Load the machine and vertices objects from the dao
        vertices = graph.vertices
        subgraph = PartitionedGraph(
            label="partitioned graph for {}".format(graph.label))

        graph_mapper = GraphMapper(graph.label, subgraph.label)

        core_constrained_vertices, chip_constrained_vertices, \
            same_chip_constrained_vertices, not_constrained_vertices = \
            self._locate_placement_constrainted_vertices(vertices)
        done_vertices = list()

        SpreaderPartitioner._partition_vertices_on_specific_core(
            core_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices)
        SpreaderPartitioner._partition_a_vertex_specific_chip(
            chip_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices)
        SpreaderPartitioner._partition_a_vertex_on_same_chip(
            same_chip_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices)
        self._partition_none_constrained(
            not_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices)

        return subgraph, graph_mapper

    @staticmethod
    def _partition_vertices_on_specific_core(
            core_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices):
        """ takes vertices which need to be on a specific core and checks if
        they can be placed on them. otherwise raises an error

        :param core_constrained_vertices:
        :param graph_mapper:
        :param graph:
        :param subgraph:
        :param done_vertices:
        :return:
        """
        for vertex in core_constrained_vertices:
            max_atom_constraint = \
                utility_calls.locate_constraints_of_type(
                    vertex.get_constraints(), PartitionerMaximumSizeConstraint)
            max_atoms = SpreaderPartitioner._locate_min(max_atom_constraint)
            if vertex.atoms > max_atoms:
                raise exceptions.PacmanPartitionException(
                    "this vertex cannot be placed / partitioned becuase the "
                    "max_atom constraint is less than the number of neurons"
                    "which would require multiple cores, yet you have "
                    "requested a speicifc core to run the entire vertex."
                    "Please adjust your constraints and try again")
            else:
                vertex_slice = Slice(0, vertex.atoms - 1)
                subvertex = vertex.create_subvertex(
                    vertex.get_resources_used_by_atoms(vertex_slice, graph),
                    label="partitioned vertex with atoms {} to {} for vertex {}"
                          .format(vertex_slice.lo_atom, vertex_slice.hi_atom,
                                  vertex.label))
                subgraph.add_subvertex(subvertex)
                graph_mapper.add_subvertex(subvertex, vertex_slice.lo_atom,
                                           vertex_slice.hi_atom, vertex)
                done_vertices.append(vertex)

    @staticmethod
    def _partition_a_vertex_specific_chip(
            chip_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices):
        """

        :param chip_constrained_vertices:
        :param graph_mapper:
        :param graph:
        :param subgraph:
        :param done_vertices:
        :return:
        """
        for vertex in chip_constrained_vertices.keys():
            if vertex not in done_vertices:
                vertex_constraint = chip_constrained_vertices[vertex]
                other_vertices_for_chip = list()
                #locate other vertices which need to reside on the same chip.
                for other_vertex in chip_constrained_vertices.keys():
                    other_constraint = chip_constrained_vertices[other_vertex]
                    if (vertex != other_vertex and
                            other_constraint.x == vertex_constraint.x and
                            other_constraint.y == vertex_constraint.y):
                        other_vertices_for_chip.append(other_vertex)


        pass

    @staticmethod
    def _partition_a_vertex_on_same_chip(
            same_chip_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices):
        """

        :param same_chip_constrained_vertices:
        :param graph_mapper:
        :param graph:
        :param subgraph:
        :param done_vertices:
        :return:
        """
        pass

    @staticmethod
    def _locate_placement_constrainted_vertices(vertices):
        """
        helper method which locates methods with a placement constraint.
        :param vertices: the partitionable graphs vertices
        :return: 4 lists where the first is vertices which are constrained to
        a processor and the second is constrained to a chip, 3rd is ones which
        need to be placed on the same chip as others and last is not constrained
        """
        none_constrainted = list()
        chip_with_processors = dict()
        chip_constrained = dict()
        same_chip = dict()
        for vertex in vertices:
            placement_constraints = utility_calls.locate_constraints_of_type(
                vertex.get_constraints(), AbstractPlacerAlgorithm)
            if len(placement_constraints) == 0:
                none_constrainted.append(vertex)
            else:
                for constraint in placement_constraints:
                    if isinstance(constraint, PlacerChipAndCoreConstraint):
                        if constraint.p is not None:
                            chip_with_processors[vertex] = constraint
                        else:
                            chip_constrained[vertex] = constraint
                    if isinstance(constraint,
                                  PlacerSubvertexSameChipConstraint):
                        same_chip[vertex] = constraint
        return chip_with_processors, chip_constrained, same_chip, \
            none_constrainted

    def _partition_none_constrained(
            self, not_constrained_vertices, graph_mapper, graph, subgraph,
            done_vertices):
        """

        :param not_constrained_vertices:
        :param graph_mapper:
        :param graph:
        :param subgraph:
        :param done_vertices:
        :return:
        """
        free_processors = self._placer_algorithm.total_free_processors()
        min_max = self._determine_min_max_atoms(not_constrained_vertices,
                                                done_vertices)
        total_atoms = 0
        for vertex in not_constrained_vertices:
            if vertex not in done_vertices:
                total_atoms += vertex.atoms
        min_per_core = math.ceil(total_atoms / free_processors)
        if min_max <= min_per_core:
            for vertex in not_constrained_vertices:
                SpreaderPartitioner._deal_with_none_constrained_vertices(
                    vertex, done_vertices, min_per_core, graph, subgraph,
                    graph_mapper)
        else:
            vertex, vertex_min = \
                self._locate_limited_vertex(not_constrained_vertices,
                                            min_per_core)
            SpreaderPartitioner._deal_with_none_constrained_vertices(
                vertex, done_vertices, vertex_min, graph, subgraph,
                graph_mapper)
            done_vertices.append(vertex)
            self._partition_none_constrained(
                not_constrained_vertices, graph_mapper, graph, subgraph,
                done_vertices)

    @staticmethod
    def _locate_limited_vertex(not_constrained_vertices, min_per_core):
        """

        :param not_constrained_vertices:
        :param min_per_core:
        :return:
        """
        for vertex in not_constrained_vertices:
            constraints = vertex.get_constraints()
            max_constraints = utility_calls.\
                locate_constraints_of_type(constraints,
                                           PartitionerMaximumSizeConstraint)
            max_values = list()
            for constrant in constraints:
                max_values.append(constrant.size)
            vertex_min = SpreaderPartitioner._locate_min(max_constraints)
            if min_per_core > vertex_min:
                return vertex, vertex_min

    @staticmethod
    def _locate_min(constraints):
        """

        :param constraints:
        :return:
        """
        min_value = None
        for constraint in constraints:
            if min_value is None:
                min_value = constraint.size
            else:
                if min_value >= constraint.size:
                    min_value = constraint.size
        return min_value

    @staticmethod
    def _deal_with_none_constrained_vertices(
            vertex, done_vertices, min_per_core, graph, subgraph, graph_mapper):
        """

        :param vertex:
        :param done_vertices:
        :param min_per_core:
        :param graph:
        :param subgraph:
        :param graph_mapper:
        :return:
        """
        if vertex not in done_vertices:
            total_atoms = 0
            while total_atoms < vertex.atoms:
                vertex_slice = None
                if total_atoms + min_per_core > vertex.atoms - 1:
                    vertex_slice = Slice(total_atoms,
                                         vertex.atoms - total_atoms)
                else:
                    vertex_slice = Slice(total_atoms,
                                         total_atoms + min_per_core)
                subvertex = vertex.create_subvertex(
                    vertex.get_resources_used_by_atoms(vertex_slice, graph),
                    label="partitioned vertex with atoms {} to "
                          "{} for vertex {}".format(vertex_slice.lo_atom,
                                                    vertex_slice.hi_atom,
                                                    vertex.label))
                subgraph.add_subvertex(subvertex)
                graph_mapper.add_subvertex(subvertex, vertex_slice.lo_atom,
                                           vertex_slice.hi_atom, vertex)
                total_atoms += min_per_core
            done_vertices.append(vertex)

    @staticmethod
    def _determine_min_max_atoms(vertices, done_vertices):
        """ helper method to determine what the min max value is based on
        the vertex max contraint.

        :param vertices: the vertexes of the application
        :return: a min max int value.
        """
        min_max = sys.maxint
        for vertex in vertices:
            if vertex not in done_vertices:
                constraints = vertex.get_constraints()
                max_constraints = utility_calls.\
                    locate_constraints_of_type(constraints,
                                               PartitionerMaximumSizeConstraint)
                max_values = list()
                for constrant in constraints:
                    max_values.append(constrant.size)
                vertex_min = SpreaderPartitioner._locate_min(max_constraints)
                min_max = min(min_max, vertex_min)
        return min_max

    def set_placer_algorithm(self, placer_algorithm, machine):
        """ setter method for setting the placer algorithm

        :param placer_algorithm: the new placer algorithm
        :type placer_algorithm: implementation of \
pacman.operations.placer_algorithms.abstract_placer_algorithm.
AbstractPlacerAlgorithm
        :param machine: the machine object
        :type machine: spinnmachine.machine.Machine object

        :return: None
        :rtype: None
        :raise PacmanConfigurationException: if the placer_algorithm is not a\
        implementation of \
pacman.operations.placer_algorithms.abstract_placer_algorithm.
AbstractPlacerAlgorithm

        """
        if issubclass(placer_algorithm, AbstractPlacerAlgorithm):
            self._placer_algorithm = placer_algorithm(machine)
        else:
            raise exceptions.PacmanConfigurationException(
                "The placer algorithm submitted is not a recognised placer "
                "algorithm")