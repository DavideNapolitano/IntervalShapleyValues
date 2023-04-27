
import itertools
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from shap import KernelExplainer
from sklearn.ensemble import RandomForestClassifier

#%%
class Hypercube:
    '''
    A class to create a hypercube object which stores values on vertices
    and values on the edges between neighboring vertices
    '''

    def __init__(self, n_vertices, vertex_keys=None, vertex_values=None):
        self.n_vertices = n_vertices
        self.V = [np.array([])] + all_subsets(n_vertices)
        self.V_value = {str(v): 0 for v in self.V}
        self.E = []
        self.E_value = {}
        self.partial_gradient = {vertex: {} for vertex in range(n_vertices)}

    def set_vertex_values(self, vertex_values):
        for v in vertex_values:
            self.V_value[v] = vertex_values[v]

        # edge values are the differences between neighboring vertex values
        self._calculate_edges()

    def _calculate_edges(self):

        # calculate the usual gradients: the difference between neighboring edges
        for i, v in enumerate(self.V):
            for _v in self.V[i + 1:]:
                if self._vertices_form_a_valid_edge(v, _v):
                    self.E.append((v, _v))
                    self.E_value[str((v, _v))] = self.V_value[str(_v)] - self.V_value[str(v)]

        # calculate partial gradients
        for vertex in range(self.n_vertices):
            self.partial_gradient[vertex] = self.E_value.copy()
            for v, _v in self.E:
                is_relevant_edge_for_partial_gradient = (vertex in v and vertex not in _v) or (
                            vertex in _v and vertex not in v)
                if not is_relevant_edge_for_partial_gradient:
                    self.partial_gradient[vertex][str((v, _v))] = 0

    def _vertices_form_a_valid_edge(self, v, _v):
        # vertices are neighbors in a hypercube
        # if they differ by exactly one element

        differ_in_size_by_1 = (abs(len(v) - len(_v)) == 1)
        the_intersection = np.intersect1d(v, _v)
        intersection_is_nonempty = len(the_intersection) > 0 or len(v) == 0 or len(_v) == 0
        v_is_the_intersection = np.all(v == the_intersection)
        _v_is_the_intersection = np.all(_v == the_intersection)

        return differ_in_size_by_1 and intersection_is_nonempty and (v_is_the_intersection or _v_is_the_intersection)


####################

def get_residual(old_cube, new_cube, vertex):
    '''
    returns: residual dictionary

        { edge : ▼_player_v[edge] - ▼v_player[edge] for edge in old_cube }
    '''
    assert set(old_cube.E_value.keys()) == set(new_cube.E_value.keys())

    res = {}
    for e in old_cube.E_value.keys():
        res[e] = old_cube.partial_gradient[vertex][e] - new_cube.E_value[e]
    return res


def residual_norm(old_cube, vertex_values, vertex):
    '''
    old_cube: v, our game
    vertex: player
    vertex_values: v_player, proposed game

    assumes that the order of the values in vertex_values align with the order of the values in old_cube.V

    returns: || ▼_player_v - ▼v_player ||
    '''
    new_cube = Hypercube(3)
    new_cube.set_vertex_values({str(_vertex): vertex_values[j] for j, _vertex in enumerate(old_cube.V)})
    return sum([abs(r) for r in get_residual(old_cube, new_cube, vertex).values()])


def all_subsets(n_elts):
    '''
        returns a list of 2^{n_elts} lists
        each a different subset of {1, 2,...,n_elts}
    '''
    res = [np.array(list(itertools.combinations(set(range(n_elts)), i))) for i in range(n_elts)]
    res = {i: res[i] for i in range(n_elts)}
    res[n_elts] = np.array([i for i in range(n_elts)]).reshape(1, -1)
    return [res[i][j] for i in range(1, n_elts + 1) for j in range(res[i].shape[0])]


#%%
# generate data
x1 = np.random.randn(500)
x2 = np.random.randn(500)
x3 = np.random.randn(500)

# label depends on interaction of X1 and X2, and not at all on X3
y = np.int0(np.sqrt(x1**2 + x2**2) < 1)
df = pd.DataFrame({"Y":y, "X1":x1, "X2":x2, "X3":x3})
features = df.iloc[:,[1,2,3]]
labels = df.iloc[:,0]

#%%
# train random forest
model = RandomForestClassifier(n_estimators=25)
model.fit(features, labels)

# train explainer on the model and the data
explainer = KernelExplainer(lambda x : model.predict_proba(x), features)

#%%
# select instance from data
instance = features.values[0,:]

# compute shapley values
shap_values = explainer.shap_values(instance)[1]

# compute expected value of model for each coalition relative to baseline (model mean)
coalition_estimated_values = {str(np.array([])): 0}
coalitions = np.array([[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
for coalition in coalitions:
    synth = pd.DataFrame(explainer.synth_data)
    for feature in coalition:
        synth = synth[synth[feature] == instance[feature]]
        model_mean = np.mean(labels)
        impact = np.mean(explainer.y[synth.index][:,1]) - model_mean
        coalition_estimated_values[str(coalition)] = impact

#%%
cube = Hypercube(3)
cube.set_vertex_values(coalition_estimated_values)

#%%
# constrained optimization:
# the null vertex must always have value 0 since it represents the empty coalition,
# but all other vertices in the new cube are subject to the minimizer
# so x0 has 7 elements instead of 8, and we always append
# a 0 to the head of the input array in the optimized functions
x0 = np.array([0.5] * 7)
f0 = lambda x: residual_norm(cube, np.append(np.array(0), x), 0)
f1 = lambda x: residual_norm(cube, np.append(np.array(0), x), 1)
f2 = lambda x: residual_norm(cube, np.append(np.array(0), x), 2)

print('solving first cube...')
v0 = minimize(f0, x0)
print('..done')

print('solving second cube...')
v1 = minimize(f1, x0)
print('..done')

print('solving third cube...')
v2 = minimize(f2, x0)
print('..done')

# residual = ||▼_feature_cube - ▼cube_feature|| after optimization
residuals = [v0.fun, v1.fun, v2.fun]

#%%
print('residuals: ', residuals)
print('shapley values: ', shap_values)