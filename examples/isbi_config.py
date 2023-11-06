""" Configuration for the tracking model """

import logging
from functools import partial

import numpy as np
from scipy.stats import beta, binom, halfnorm, norm
from tensor_tree.impl_np import NP_Impl

from uat.assignment import (
    SimpleContinueGenerator,
    SimpleEndTrackAssGenerator,
    SimpleNewAssGenerator,
    SimpleSplitGenerator,
)
from uat.models import (
    ConstantModel,
    ModelExecutor,
    area_growth_computer,
    distance_to_pred_computer,
    distance_to_pred_masked,
    split_dist_computer,
)
from uat.utils import ContourDistanceCache, NearestNeighborCache

backend = NP_Impl()


def split_pair_filter(
    source_index, target_index, cdc: ContourDistanceCache, df, distance_threshold=1.0
):
    """Make sure that distances for splitting are not too large"""
    # pylint: disable=unused-argument
    if len(target_index) == 0:
        return np.zeros((len(target_index)), dtype=bool)
    mask = cdc.distance(target_index[:, 0], target_index[:, 1]) < distance_threshold

    return mask


def create_split_children_distance_model(
    data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=10)
):
    # pylint: disable=unused-argument
    def split_distance_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        """Extracts the distance between the children of proposed assignments

        Args:
            tracking (_type_): tracking lineage
            source_index (_type_): numpy array of sources for assignments
            target_index (_type_): numpy array of targets for assignments
            df (_type_): dictionary with indexed information

        Returns:
            _type_: euclidean distance between the children of the assignments
        """
        major_extents = df["major_extents"]

        distances = np.zeros(len(target_index), dtype=np.float32)

        for i, (a, b) in enumerate(target_index):
            loc_distances = np.linalg.norm(
                major_extents[a][:, None, :] - major_extents[b][None, :, :], axis=-1
            )

            distances[i] = np.min(loc_distances)

        return distances

    split_children_distance_model = ModelExecutor(
        split_distance_extractor,  # extract the distances
        prob,  # compute the probability
        data,
    )

    return split_children_distance_model

def compute_angle_between_children(maj_extA, maj_extB):
    """Computes the angle between the children

        uses the major axes and the poles in order to orient them accordingly.

        The angle is computed in degrees [0...180]
    """

    loc_distances = np.linalg.norm(
                maj_extA[:, None, :] - maj_extB[None, :, :], axis=-1
            )
    
    min_distance = np.min(loc_distances)
    #print(np.array(np.where(min_distance == loc_distances)).flatten())
    
    # use the directions pointing towads the new tips
    data = np.array(np.where(min_distance == loc_distances))
    #print(data.shape)
    #print(loc_distances)
    a_ind, b_ind = data[:,0].flatten()
    adir =  maj_extA[1-a_ind] - maj_extA[a_ind]
    bdir =  maj_extB[1-b_ind] - maj_extB[b_ind]
    #print(adir, bdir)
    
    return np.arccos(np.dot(adir, bdir) / np.linalg.norm(adir) / np.linalg.norm(bdir)) * 180 / np.pi

def log_sum(la,lb):
    """log(a + b) = la + log1p(exp(lb - la)) from https://stackoverflow.com/questions/65233445/how-to-calculate-sums-in-log-space-without-underflow"""
    #la = np.log(a)
    #lb = np.log(b)
    return la + np.log1p(np.exp(lb - la))

def prob_angles(angles, loc, scale):

    # compute the difference
    diff = np.abs(loc - angles)

    # compute the probability of more extreme values
    prob = log_sum(norm.logcdf(loc - diff, loc=loc, scale=scale), norm.logsf(loc + diff, loc=loc, scale=scale))

    # do not use nan values
    prob[np.isnan(prob)] = -30

    return prob

def prob_cont_angles(angles, scale):

    # compute the probability of more extreme values
    prob = halfnorm.logsf(angles, loc=scale)#log_sum(norm.logcdf(loc - diff, loc=loc, scale=scale), norm.logsf(loc + diff, loc=loc, scale=scale))

    # do not use nan values
    prob[np.isnan(prob)] = -30

    return prob

def create_continue_angle_model(
    data,
    prob
):
    # pylint: disable=unused-argument
    def continue_angle_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        major_axis = df["major_axis"]

        majA = major_axis[source_index.flatten()]
        majB = major_axis[target_index.flatten()]

        norm = np.linalg.norm(majA, axis=-1) / np.linalg.norm(majB, axis=-1)

        raw_values = np.stack([
            [np.dot(a, b) for a,b in zip(majA, majB)] / norm,
            [np.dot(-a, b) for a,b in zip(majA, majB)] / norm,
        ], axis=-1)

        raw_values = np.clip(raw_values, -1., 1.)

        angles = np.min(np.arccos(raw_values) * 180 / np.pi, axis=-1)

        return angles

    continue_angle_model = ModelExecutor(continue_angle_extractor, prob, data)

    return continue_angle_model

def create_split_children_angle_model(
    data,
    prob=prob_angles
#    lambda vs: norm.logpdf(
#        np.abs(np.cos(vs * (2 * np.pi / 360))),
#        loc=np.cos(0 * 2 * np.pi / 360),
#        scale=0.1,
#    ),
):
    # pylint: disable=unused-argument
    def split_angle_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        major_extents = df["major_extents"]

        cos_angles = np.zeros(len(target_index), dtype=np.float32)

        for i, (a, b) in enumerate(target_index):
            # extract both major axes vectors
            majA = major_extents[a]
            majB = major_extents[b]

            #print(majA)
            #print(majB)

            cos_angles[i] = compute_angle_between_children(majA, majB)

        return cos_angles

    split_children_angle_model = ModelExecutor(split_angle_extractor, prob, data)

    return split_children_angle_model


def create_split_rate_model(data):
    def split_rate_comp(values):
        probs = binom.logpmf(values + 1, n=40, p=10 / 40)
        indices = np.where(values == -1)
        probs[indices] = np.log(
            0.5
        )  # 50% split probability when we have no parents available
        print(probs)
        return probs

    # pylint: disable=unused-variable
    split_rate_model = ModelExecutor(
        split_dist_computer,
        # continue_rate_model.model.invert(),
        split_rate_comp,
        data,
    )

    return split_rate_model


def growth_probs(values, mean=1.06, scale=0.3):
    probs = halfnorm.logsf(np.abs(mean - values), scale=scale)
    return probs


def create_growth_model(data, mean, scale):
    return ModelExecutor(
        area_growth_computer,
        # growth_probs,
        partial(growth_probs, mean=mean, scale=scale),
        data,
    )


def distance_to_previous(tracking, source_index, target_index, data):
    # pylint: disable=unused-argument
    all_centroids = data[
        "centroid"
    ]  # np.array(data['centroid'].to_list(), dtype=np.float32)

    sources = all_centroids[source_index]
    targets = all_centroids[target_index]

    distances = np.linalg.norm(sources - targets, axis=-1)

    #if len(distances.shape) == 2:
    # take the sum of distances (only one distance for migration, two distances for division)
    distances = np.sum(distances, axis=-1)

    return distances


def create_continue_keep_position_model(data, prob):
    return ModelExecutor(
        # partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        distance_to_previous,
        prob,
        #lambda values: beta.logsf(
        #    np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        #),
        # lambda values: expon.logsf(values, scale=5),
        data,
    )

def dis_app_prob_func(x_coordinates, max_overshoot, min_dist, x_size: int):
    total_auc = 0.5 * (min_dist + max_overshoot)**2
    area_func = lambda x: 0.5 * (x + max_overshoot) ** 2
    prob = np.log(1-area_func(np.clip(np.min(np.stack([x_coordinates, x_size - x_coordinates]), axis=0), -max_overshoot, min_dist)) / total_auc)

    prob[np.isnan(prob)] = -30

    return prob


def source_position_extractor(
    tracking,
    source_index,
    target_index,
    df,
):
    # extract the x-position of all sources
    return df["centroid"][source_index.flatten()][:, 1]


def create_disappear_model(data, width: int):
    return ModelExecutor(
        source_position_extractor,
        partial(dis_app_prob_func, max_overshoot=20, min_dist=20, x_size=width),
        data
    )


def create_continue_temp_position_model(data, prob, history=3):
    return ModelExecutor(
        partial(distance_to_pred_computer, alpha=None, history=history, stop_at_split=True),
        prob,
        data,
    )


def create_split_movement_model(data, prob):
    return ModelExecutor(
        #partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        distance_to_previous,
        prob,
        #lambda values: beta.logsf(
        #    np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        #).sum(axis=-1),
        # lambda values: expon.logsf(values, scale=5).sum(axis=-1),
        data,
    )

def create_end_pos_model(data, width):
    def position(tracking, source_index, target_index, df):
        # pylint: disable=unused-argument
        all_positions = df["centroid"]

        return all_positions[source_index.reshape(-1)]
    
    def pos_log_pdf(values):
        values = values[:, 0]
        min_dist = np.minimum(values, width - values)

        p = np.zeros_like(values)
        p[min_dist > 60] = -1  # -6 #np.log(1e-6)

        return p
    
    return ModelExecutor(position, pos_log_pdf, data)

class SimpleCDC:
    """class to compute contour distances (approximated by center position in favor of speed)"""

    def __init__(self, positions):
        self.positions = positions

    def distance(self, indA, indB):
        return np.linalg.norm(self.positions[indA] - self.positions[indB], axis=1)


def setup_assignment_generators(df, data, width, subsampling, use_split_distance = False, use_split_angle = False, mig_growth_scale=0.05, mig_movement_scale=20, div_growth_scale=None, div_movement_scale=20, app_prob = 0.25):

    if div_growth_scale is None:
        div_growth_scale = 2 * mig_growth_scale
    if div_movement_scale is None:
        div_movement_scale = 2 * mig_movement_scale

    # TODO: Why these values?
    constant_new_model = ConstantModel(np.log(app_prob))
    #constant_end_model = ConstantModel(np.log(app_prob))
    constant_end_model = create_disappear_model(data, width)

    end_pos_model = create_end_pos_model(data, width)

    positions = data["centroid"]

    cdc = SimpleCDC(positions)

    logging.info("Compute nearest neighbor cache")
    nnc = NearestNeighborCache(df)


    filters = [lambda s, t: nnc.kNearestNeigborsMatrixMask(15, s, t)]

    # def sf_debug(s,t):
    #    print(s,t)

    split_filters = []  # [sf_debug]

    max_distance = 200

    #continue_keep_position_model = create_continue_keep_position_model(
    #    data, max_distance
    #)

    def movement_log_pdf(values):

        p = halfnorm.logsf(x=values, scale=5)  # * 3

        p[values.mask] = 0.0

        return p

    # pylint: disable=unused-variable
    continue_keep_speed_model = ModelExecutor(
        distance_to_pred_masked, movement_log_pdf, data
    )

    # create the movement models
    continue_keep_position_model = create_continue_keep_position_model(
        data, prob=lambda val: halfnorm.logsf(val, scale=mig_movement_scale*subsampling)
    )

    continue_keep_position_model = create_continue_temp_position_model(
        data, prob=lambda val: halfnorm.logsf(val.flatten(), scale=mig_movement_scale*subsampling)
    )

    split_movement_model = create_split_movement_model(data, prob=lambda val: halfnorm.logsf(val, scale=div_movement_scale*subsampling))

    # pylint: disable=unused-variable
    split_rate_model = create_split_rate_model(data)

    # create distance models for splits
    split_children_distance_model = create_split_children_distance_model(data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=3*subsampling))

    continue_angle_model = create_continue_angle_model(data, prob=partial(prob_cont_angles, scale=20*subsampling))
    split_children_angle_model = create_split_children_angle_model(data,prob=partial(prob_angles, loc=135, scale=20*subsampling))

    # create growth models
    continue_growth_model = create_growth_model(data, mean=1.008**subsampling, scale=mig_growth_scale * subsampling)
    split_growth_model = create_growth_model(data, mean=1.016**subsampling, scale=div_growth_scale * subsampling)

    migration_models = [
        continue_keep_position_model,
        # continue_keep_speed_model,
        continue_growth_model,
    ]
    split_models = [
        split_movement_model,
        # split_rate_model,
        split_growth_model,
        #split_children_distance_model,
        #split_children_angle_model,
    ]  # , split_distance_ratio]

    if use_split_distance:
        split_models.append(split_children_distance_model)

    if use_split_angle:
        #split_models.append(split_children_angle_model)
        #migration_models.append(continue_angle_model)
        pass

    assignment_generators = [
        SimpleNewAssGenerator([constant_new_model]),
        SimpleEndTrackAssGenerator([constant_end_model]), #, end_pos_model]),
        # SimpleContinueGenerator(filters, [continue_growth_model, continue_rate_model, continue_movement_model]),
        SimpleContinueGenerator(filters, migration_models),
        # SimpleSplitGenerator(filters, partial(split_pair_filter, cdc=cdc), [split_rate_model_simple, split_growth_model, split_sum_growth_model, split_movement_model, split_dist_prop_model]) #, split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
        SimpleSplitGenerator(
            filters + split_filters,
            partial(split_pair_filter, cdc=cdc, distance_threshold=max_distance, df=df),
            split_models,
        ),  # , split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
    ]

    return assignment_generators
