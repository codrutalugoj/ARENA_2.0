# %%

import os 
import sys
from pathlib import Path

import torch as t 
from torch import Tensor
import einops

from ipywidgets import interact
import plotly.express as px
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''

    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, steps=num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1

    return rays

rays1d = make_rays_1d(9, 10.0)
print(rays1d)

if MAIN:
    fig = render_lines_with_plotly(rays1d)


# %%
if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

# %%
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

render_lines_with_plotly(rays1d, segments)
# %%
@jaxtyped
def intersect_ray_1d(ray: Float[Tensor, "points=2 dims=3"], segment: Float[Tensor, "points=2 dims=3"]) -> Bool[Tensor, ""]:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points; O - origin, D - direction
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    (note: Bool[Tensor, ""] is for a single scalar value)
    Return True if the ray intersects the segment.
    '''

    O, D = ray[..., :2]
    L1, L2 = segment[..., :2]

    mat = t.stack([D, L1 - L2], dim=-1)    
    vec = L1 - O

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    u, v = sol[0].item(), sol[1].item()

    return (u >= 0) and (v >= 0) and (v <= 1)


intersect_ray_1d(rays1d[0], segments[0])


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


# %%
@jaxtyped
def my_concat(x: Float[Tensor, "a1 b"], y: Float[Tensor, "a2 b"]) -> Float[Tensor, "a1+a2 b"]:
    return t.concat([x, y], dim=0)


# %%
@jaxtyped
def intersect_rays_1d(rays: Float[Tensor, "nrays num_points=2 n_dims=3"], segments: Float[Tensor, "nsegments num_points=2 n_dims=3"]) -> Bool[Tensor, "nrays"]:
    '''
    Returns true for each ray if it intersects *any* segment. 

    rays: shape (num_rays, num_points=2, n_dims=3); O, D
    segments: shape (num_segments, num_points=2, n_dims=3); L1, L2
    '''

    # get 1 ray, see if it intersects any segment
    rays_repeated = einops.repeat(rays, "rays points dims -> rays segments points dims", segments=segments.shape[0])
    segments_repeated = einops.repeat(segments, "segments points dims -> rays segments points dims", rays=rays.shape[0])
    
    rays_repeated = rays_repeated[..., :2]
    segments_repeated = segments_repeated[..., :2]

    lhs_mat = t.stack([rays_repeated[:, :, 1, :], segments_repeated[:, :, 0, :] - segments_repeated[:, :, 1, :]], dim=-1)
    rhs_vec = segments_repeated[:, :, 0, :] - rays_repeated[:, :, 0, :]

    # problem: singular matrices don't have an inverse so you can't find a sol to the system
    singular_mat = t.linalg.det(lhs_mat).abs() < 1e-6

    lhs_mat[singular_mat] = t.eye(2) # set the singular matrices to [[1 0] [0 1]] = identity matrix

    sols = t.linalg.solve(lhs_mat, rhs_vec)
    u, v = sols[..., 0], sols[..., 1]

    return ((u >= 0) & (v <= 1) & (v >= 0) & ~singular_mat).any(dim=-1)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    num_rays = num_pixels_y*num_pixels_z

    rays_2d = t.zeros((num_rays, 2, 3), dtype=t.float32)
    rays_y = einops.repeat(t.linspace(-y_limit, y_limit, steps=num_pixels_y), "y_dim -> y_dim z_dim", z_dim=num_pixels_z)
    rays_z = einops.repeat(t.linspace(-z_limit, z_limit, steps=num_pixels_z), "z_dim -> y_dim z_dim", y_dim=num_pixels_y)

    rays_2d[:, 1, 0] = 1

    rays_2d[:, 1, 1] = rays_y.reshape(num_rays)
    rays_2d[:, 1, 2] = rays_z.reshape(num_rays)

    return rays_2d


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    print(rays_2d)
    render_lines_with_plotly(rays_2d)

# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)


# %%
Point = Float[Tensor, "points=3"]

@jaxtyped
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    # 1. make the matrix of equations
    mat = t.stack([-D, B-A, C-A], dim=-1)
    vec = O - A

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    s, u, v = sol

    return (u >= 0) and (v >= 0) and ((u + v) <= 1)


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)


# %%
@jaxtyped
def raytrace_triangle(rays: Float[Tensor, "nrays rayPoints=2 dims=3"], 
                      triangle: Float[Tensor, "trianglePoints=3 dims=3"]
                      ) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    nrays = rays.shape[0]

    # intersections = t.zeros((nrays))

    triangle_repeated = einops.repeat(triangle, "trianglePoints dims -> nrays trianglePoints dims", nrays=nrays)

    A, B, C = triangle_repeated.unbind(dim=1)
    assert A.shape == (nrays, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == (nrays, 3)

    mat = t.stack([-D, B-A, C-A], dim=-1)
    vec = O - A

    zero_det = t.linalg.det(mat).abs() < 1e-8

    mat[zero_det] = t.eye(rays.shape[2])

    sols = t.linalg.solve(mat, vec)
    s, u, v = sols.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~zero_det)



if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    # render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)

    print(intersects)

    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
### Incorrect function debugging

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    # orig: NR = rays.size[0]
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    # orig: O, D = rays.unbind(-1)
    O, D = rays.unbind(1)

    # orig: mat = t.stack([- D, B - A, C - A]) 
    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

def raytrace_mesh(rays: Float[Tensor, "nrays npoints=2 dims=3"], 
                  triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
                  ) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    distances = t.zeros((nrays)) + t.inf

    rays = einops.repeat(rays, "nrays npts dims -> nrays ntriangles npts dims", ntriangles=ntriangles)
    triangles = einops.repeat(triangles, "ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims", nrays=nrays)

    print(rays.shape, triangles.shape)
    
    A, B, C = triangles.unbind(dim=2)
    O, D = rays.unbind(dim=2)

    mat = t.stack([-D, B-A, C-A], dim=-1)
    vec = O - A

    zero_det = t.linalg.det(mat).abs() < 1e-8

    mat[zero_det] = t.eye(rays.shape[3])

    sols = t.linalg.solve(mat, vec)

    s, u, v = sols.unbind(dim=-1)
    assert u.shape == (nrays, ntriangles)

    # get intersections
    intersections = (u >= 0) & (v >= 0) & (u + v <= 1) & ~zero_det

    # set the distances to inf for rays that don't intersect any triangle
    s[~intersections] = t.inf

    return t.min(s, dim=1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    print((dists == t.inf).any())

    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()

# TODO: Bonus exercises