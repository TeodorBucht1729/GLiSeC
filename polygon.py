
import os
import random
import pyclipper
import matplotlib.pyplot as plt
import numpy as np
import limit_calc as lc
import pickle
import copy
import time
from multiprocessing import Process

nbr_intersections = 0
  

def plot_polygons(polygons, title, scale_factor=2**31, complex_points = None, check_points_inside = True,  save_name=None, xlim=None, ylim=None, poly_color=None, poly_opacity=1, reverse_poly_color = None):
    fig, ax = plt.subplots()
    polygons = polygons.copy()
    if len(polygons) > 0 and type(polygons[0]) == tuple:
        polygons = [polygons] 
    clipper_polygons = pyclipper.scale_to_clipper(polygons, scale_factor)
    for poly_ind, polygon in enumerate(polygons):
        polygon.append(polygon[0])
        xs, ys = zip(*polygon)
        if poly_color is not None:
            if reverse_poly_color is not None:
                area = pyclipper.Area(clipper_polygons[poly_ind])
                # print("Area:", area)
                if area >= 0:
                    ax.plot(xs, ys, poly_color, alpha=poly_opacity)
                else:
                    ax.plot(xs, ys, reverse_poly_color, alpha=poly_opacity)
            else:
                ax.plot(xs, ys, poly_color, alpha=poly_opacity)
        else:
            ax.plot(xs, ys, alpha=poly_opacity)

    if complex_points is not None and check_points_inside and len(complex_points) > 0:
        clipper_poly = pyclipper.scale_to_clipper(polygons, scale_factor)
        x = complex_points.real
        y = complex_points.imag
        points_in = []
        points_out = []
        for i in range(len(x)):
            point_in = max((pyclipper.PointInPolygon((x[i], y[i]), polygon) for polygon in clipper_poly))   
            if point_in >= 0:
                points_in.append((x[i], y[i]))
            else:
                points_out.append((x[i], y[i]))

        if len(points_in) > 0:
            plot_points(points_in, ax, 'b')
        if len(points_out) > 0:
            plot_points(points_out, ax, 'r')
        print("Number of points inside polygon:", len(points_in), "\nNumber of points outside polygon:", len(points_out))
    elif complex_points is not None and len(complex_points) > 0:
        xs = complex_points.real
        ys = complex_points.imag
        points = [(xi, yi) for xi, yi in zip(xs, ys)]
        plot_points(points, ax, 'lime')

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.ylabel('$\Im \lambda$')
    plt.xlabel('$\Re \lambda$')
    plt.title(title)
    ax.set_aspect('equal')
    if save_name is not None:
        plt.savefig(save_name)
    return fig, ax


def plot_points(points, ax, mods):
    xs, ys = zip(*points)
    ax.scatter(xs, ys, color=mods, s=16)

def moving_average(ys, window_size):
    assert window_size % 2 == 1, "window size needs to be odd"
    y = ys.copy()
    smoothed = y.copy()
    y = np.pad(y, window_size//2)
    part_sum = np.sum(y[0:window_size-1])
    for i in range(window_size//2, len(ys)+window_size//2):
        part_sum += y[i+window_size//2]
        smoothed[i-window_size//2] = part_sum/window_size
        part_sum -= y[i-window_size//2]
    return smoothed

def find_rho_limits(a: np.poly1d, r, s, sample_points, marigin):
    K = marigin*np.max(np.abs(a(np.exp(np.linspace(0, 2 * np.pi, sample_points) * 1j))))
    coeffs = a.c
    abs_pol = [np.abs(a_n) for a_n in coeffs]
    lo_bound = abs_pol.copy()
    lo_bound[-1] *= -1
    lo_bound[s] += K
    real_roots = [np.real(x) for x in np.roots(lo_bound) if np.imag(x) == 0 and np.real(x) > 0] 
    lo = min(real_roots)
    hi_bound = abs_pol.copy()
    hi_bound[0] *= -1
    hi_bound[s] += K
    real_roots = [np.real(x) for x in np.roots(hi_bound) if np.imag(x) == 0 and np.real(x) > 0] 
    hi = max(real_roots)
    return (lo, hi)

def generate_rhos(rho_lo, rho_hi, nbr_rhos, equidistant = True):

    if equidistant:
        return np.linspace(rho_lo, rho_hi ,nbr_rhos)
    else:
        if rho_lo <= 1 and 1 <= rho_hi:
            delta = (rho_hi + 1/rho_lo - 2)/(nbr_rhos-1)
            l = int(np.ceil(1/delta * (1/rho_lo - 1)))
            b = int(np.ceil((rho_hi-1)/delta))

            rhos = np.flip(1/(1 + delta * np.linspace(1, l, l)))
            rhos = np.append(rhos, 1 + delta*np.linspace(0, b, b+1))
        elif rho_lo < 1 and rho_hi < 1:
            delta = (1/rho_lo - 1/rho_hi)/(nbr_rhos - 1)
            l = int(np.ceil(1/delta * (1/rho_lo - 1)))
            b = int(np.floor(1/delta * (1/rho_hi - 1)))
            rhos = np.flip(1/(1 + delta * np.linspace(b, l, l-b+1)))
        elif rho_lo >= 1 and rho_hi >= 1:
            delta = (rho_hi - rho_lo)/(nbr_rhos - 1)
            l = int(np.floor(1/delta * (rho_lo - 1)))
            b = int(np.ceil(1/delta * (rho_hi - 1)))
            rhos = 1 + delta*np.linspace(l, b, b-l+1)

        return rhos

def eval_laurent(a: np.poly1d, r, s, z):
    ret = 0
    for n in range(-r, s+1):
        ret += a.coef[-(n+r+1)] * z**n
    return ret

def intersect_polygon(a: np.poly1d, r, s, sample_points, curr_poly, rho, scale_factor, delta = None, smart_delta = False):

    global nbr_intersections
    nbr_intersections += 1
    start_area = sum((abs(pyclipper.Area(poly)) for poly in curr_poly))
    a_disc_values = []
    zds = []
    for sample_point in sample_points:
        z = rho*np.exp(1.0j * sample_point)
        a_disc_values.append(eval_laurent(a, r, s, z))
        zds.append(z)
    zds = np.array(zds)
    a_disc = [(np.real(az), np.imag(az)) for az in a_disc_values]
    pc = pyclipper.Pyclipper()
    pc.AddPaths(curr_poly, pyclipper.PT_SUBJECT, True)
    if delta is not None:
        # offseting intersection 
        if smart_delta:
          
            second_deriv = np.zeros(len(sample_points), dtype='complex128')
            n = -r
            coeffs = (a.c).astype('complex128')
            while n <= s:
                second_deriv += (n + n*(n-1))* np.power(zds, n) * coeffs[-(n+r+1)]
                n += 1
            delta = (2*np.pi/(len(sample_points)))**2 * (np.max(np.abs(np.real(second_deriv))) + np.max(np.abs(np.imag(second_deriv))))
            # add some margin for error
            delta *= 1.2
             

        pco = pyclipper.PyclipperOffset()
        other_poly = pyclipper.scale_to_clipper(a_disc, scale_factor)
        other_poly = pyclipper.SimplifyPolygons([other_poly], pyclipper.PFT_NONZERO)
        pco.AddPaths(other_poly, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        other_poly = pco.Execute(delta*scale_factor)
        poly_i = 0
        one_path_clip = []
        while poly_i < len(other_poly):
            one_path_clip.extend(other_poly[poly_i])
            one_path_clip.append(other_poly[poly_i][0])
            poly_i += 1
        poly_i = len(other_poly) - 2
        while poly_i >= 1:
            one_path_clip.append(other_poly[poly_i][0])
            poly_i -= 1
        # if rho > 1.0:
        #     plot_poly = []
        #     plot_poly.extend(other_poly)
        #     plot_poly.extend([pyclipper.scale_to_clipper(a_disc, scale_factor)])
        #     plot_polygons(plot_poly, "offset polygon")

        pc.AddPath(one_path_clip, pyclipper.PT_CLIP, True)
        result = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        return result

    pc.AddPath(pyclipper.scale_to_clipper(a_disc, scale_factor), pyclipper.PT_CLIP, True)
    intersection = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    intersection = pyclipper.SimplifyPolygons(intersection, pyclipper.PFT_NONZERO)
    end_area = sum((abs(pyclipper.Area(poly)) for poly in intersection))
    return (intersection, start_area - end_area) 

def area_search(a, r, s, rhos, rhos_indices, areas, sample_points, intersection, scale_factor):
    for rho, ind in zip(rhos, rhos_indices):
        area_test, area_diff = intersect_polygon(a, r, s, sample_points, intersection, rho, scale_factor)
        areas[ind] = area_diff
    return areas

def count_nbr_vertices(polygon):
    ans = 0
    for part_poly in polygon:
        ans += len(part_poly)
    return ans


def find_limit_set(a: np.poly1d, r, s, nbr_sample_points, nbr_rhos, manual_rho_boundary = False, rho_lo = 0.1, rho_hi = 10, scale_factor=2**31, smoothing_window_size=7, nbr_sweeps=2, plot_area_sweeps = False, threshold = 1000000, nbr_area_sweep_points = 500, equidistant_rho = True, plot_nbr_vertices = False, nbr_vertices_save_name = None, outer_bound = False, return_intermediate_steps = False):

    if return_intermediate_steps:
        assert outer_bound, "Need to compute approximating superset for return_intermediate_steps to work"

    # find biggest_rho_bounds if not specified
    if not manual_rho_boundary:
        rho_lo, rho_hi = find_rho_limits(a, r, s, nbr_sample_points, 1)
        print("Automatic rho limits generated, boundaries are", rho_lo, rho_hi)

    rhos = generate_rhos(rho_lo, rho_hi, nbr_rhos//nbr_sweeps, equidistant=equidistant_rho)
    area_check_rhos = generate_rhos(rho_lo, rho_hi, nbr_area_sweep_points, equidistant=equidistant_rho)

    # generate sample points
    sample_points = np.linspace(0, 2*np.pi, nbr_sample_points)
    sample_points = sample_points[:-1]

    # plot average a_derivative
    # plot_average_derivative(a, r, s, sample_points, area_check_rhos)

    delta_offset = 2*np.pi / nbr_sample_points

    # generate start polygon
    a_disc_values = []
    for sample_point in sample_points:
        z = rhos[0]*np.exp(1.0j * sample_point)
        a_disc_values.append(eval_laurent(a, r, s, z))
    start_polygon = [(np.real(az), np.imag(az)) for az in a_disc_values]
    start_polygon = [pyclipper.scale_to_clipper(start_polygon, scale_factor)]
    intersection = pyclipper.SimplifyPolygons(start_polygon, pyclipper.PFT_NONZERO)
    if outer_bound:
        pco = pyclipper.PyclipperOffset()
        pco.AddPaths(intersection, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        offset_polygon = pco.Execute(delta_offset)
    

    # initialize the area diffs of the sweeping rhos
    areas = np.array([-1] * len(area_check_rhos), dtype='float')

    nbr_vertices = [count_nbr_vertices(intersection)]

    if return_intermediate_steps:
        times = []
        approx_polys = []
        approx_supersets = []


    # begin sweeps
    for i_sweep in range(nbr_sweeps):
        start = time.time()
        # perform intersections
        for rho in rhos:
            intersection, area_diff = intersect_polygon(a, r, s, sample_points, intersection, rho, scale_factor)
            nbr_vertices.append(count_nbr_vertices(intersection))

            if outer_bound:
                offset_polygon = intersect_polygon(a, r, s, sample_points, offset_polygon, rho, scale_factor, delta = delta_offset, smart_delta=True)
        
        # do area checks
        if areas[0] < 0:
            # no prior area sweeps have been made
            area_search(a, r, s, area_check_rhos, list(range(len(area_check_rhos))), areas, sample_points, intersection, scale_factor)
        elif i_sweep < nbr_sweeps-1:
            # prior area sweeps have been made
            new_max = -1
            # update area diffs for good ranges.
            for good_range in good_ranges:
                area_search(a, r, s, area_check_rhos[good_range[0]:good_range[1]+1], list(range(good_range[0], good_range[1]+1)), areas, sample_points, intersection, scale_factor)
                new_max = max(np.max(areas[good_range[0]:good_range[1]+1]), new_max)

            # if we sweep bad ranges again, the resulting area diffs will always be less, so we can save time
            def in_some_range(x, ranges):
                for r in ranges:
                    if r[0] <= x and x <= r[1]:
                        return True
                return False

            search_anyway = []
            for i in range(len(area_check_rhos)):
                if not in_some_range(i, good_ranges) and new_max < areas[i]/threshold:
                    search_anyway.append(i)
            area_search(a, r, s, area_check_rhos[search_anyway], search_anyway, areas, sample_points, intersection, scale_factor)
            

        smoothed_areas = moving_average(areas, smoothing_window_size)

        # extract good ranges of rho
        good_ranges = np.where(np.diff(smoothed_areas > np.max(smoothed_areas)/threshold, prepend=0, append=0))[0].reshape(-1, 2)
        good_ranges[:, 1] -= 1
        for i in range(len(good_ranges)):
            good_ranges[i][0] = int(max(0, good_ranges[i][0] - 1))
            good_ranges[i][1] = int(min(len(area_check_rhos), good_ranges[i][1] + 1))

        # print what good ranges were found
        good_rho_ranges = []
        for good_range in good_ranges:
            good_rho_ranges.append((area_check_rhos[good_range[0]], area_check_rhos[good_range[1]]))
        print("Sweep", i_sweep, "found good ranges", good_rho_ranges)

        # generate new "good rhos"
        new_rhos = np.array([], dtype=np.longdouble)
        for good_range in good_ranges:
            extra_forward = random.random() * (area_check_rhos[good_range[0]+1] - area_check_rhos[good_range[0]])
            extra_back = random.random() * (area_check_rhos[good_range[1]] - area_check_rhos[good_range[1]-1])
            new_rhos = np.append(new_rhos, generate_rhos(area_check_rhos[good_range[0]] + extra_forward, area_check_rhos[good_range[1]] - extra_back, nbr_rhos//(nbr_sweeps * len(good_ranges)), equidistant=equidistant_rho))
        rhos = new_rhos

        end = time.time()
        if return_intermediate_steps:
            times.append(end-start)

            approx_poly = pyclipper.scale_from_clipper(intersection, scale_factor)
            approx_poly = [[(x[0], x[1]) for x in polygon] for polygon in approx_poly]
            approx_polys.append(approx_poly)

            approx_superset = pyclipper.scale_from_clipper(offset_polygon, scale_factor)
            approx_superset = [[(x[0], x[1]) for x in polygon] for polygon in approx_superset]
            approx_supersets.append(approx_superset)

        if plot_area_sweeps:
            plt.figure()
            plt.plot(area_check_rhos, smoothed_areas, '-o')
            plt.yscale('log')
            plt.xscale('log')
            plt.title("Reduced area after intersection with different rho")

    if plot_nbr_vertices:
        fig, ax = plt.subplots()
        ax.plot(nbr_vertices)
        plt.xlabel("#intersections")
        plt.ylabel("#vertices")
        plt.title("Growth of vertices of $\Lambda$")
        plt.grid(True)
        if nbr_vertices_save_name is not None:
            plt.savefig(nbr_vertices_save_name)
        # ax.set_aspect('equal')


    ret = pyclipper.scale_from_clipper(intersection, scale_factor)
    ret = [[(x[0], x[1]) for x in polygon] for polygon in ret]

    if return_intermediate_steps:
        return (times, approx_polys, approx_supersets)

    if outer_bound:
        ret_bound = pyclipper.scale_from_clipper(offset_polygon, scale_factor)
        ret_bound = [[(x[0], x[1]) for x in polygon] for polygon in ret_bound]
        return (ret, ret_bound)

    return ret


def join_polys(polys, scale_factor=2**31):
 
    pc = pyclipper.Pyclipper()

    polys = pyclipper.scale_to_clipper(polys, scale_factor)

    res=[]
    pc.AddPaths(polys, pyclipper.PT_SUBJECT, True)
    clip_polys = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    
    clip_polys = pyclipper.scale_from_clipper(clip_polys, scale_factor)
    res.extend([cp for cp in clip_polys])
    return res

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def bound_contained(inner_points, outer_bound, r_extend, scale_factor=2**31):

    nbr_vert = 20
    expanded_polygon = []
    for point in inner_points:
        poly = []
        for n in range(nbr_vert):
            poly.append((point[0] + r_extend * np.cos(2*np.pi * n/nbr_vert), point[1] + r_extend * np.sin(2*np.pi * n/nbr_vert)))
        expanded_polygon.append(poly)
    merged_poly = join_polys(expanded_polygon)

    # intersect merged_poly and outer_bound
    pc = pyclipper.Pyclipper()
    outer_bound = pyclipper.scale_to_clipper(outer_bound, scale_factor)
    merged_poly = pyclipper.scale_to_clipper(merged_poly, scale_factor)
    pc.AddPaths(outer_bound, pyclipper.PT_CLIP, True)
    pc.AddPaths(merged_poly, pyclipper.PT_SUBJECT, True)
    intersection = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    # if the intersection is precisecly the outer bound, outer bound is cointained in merged_poly
    if intersection == outer_bound:
        return True
    else:
        return False


def error_bound(inner_points, outer_bound, hi_manual = None):

    poly_points = []
    for poly in outer_bound:
        poly_points.extend(poly)
    if hi_manual is None:
        hi = max((dist2(inner_points[0], p) for p in poly_points))**0.5
    else:
        hi = hi_manual
    lo = 0

    while (hi - lo) > 10**-6:
        print("lo hi:", lo, hi)
        mid = (hi + lo)/2
        if bound_contained(inner_points, outer_bound, mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def bound_contained_known_example(order, outer_bound, r_extend, scale_factor=2**31):

    nbr_vert = 20
    r = order-1
    R = (r+1) * r**(-r/(r+1))

    expanded_polygon = []
    for k in range(order):
        poly = []
        eps = np.cos(2*np.pi / order * k) + 1j * np.sin(2*np.pi / order * k)
        for rot in range(nbr_vert+1):
            vrt_rot = np.cos(np.pi / nbr_vert * rot) + 1j * np.sin(np.pi / nbr_vert * rot)
            vrt_rot *= (-1j* eps)
            poly.append(((vrt_rot*r_extend + R * eps).real, (vrt_rot*r_extend + R * eps).imag))
        for rot in range(nbr_vert+1):
            vrt_rot = np.cos(np.pi / nbr_vert * rot) + 1j * np.sin(np.pi / nbr_vert * rot)
            vrt_rot *= (1j* eps)
            poly.append(((vrt_rot*r_extend).real, (vrt_rot*r_extend).imag))
        expanded_polygon.append(poly)
    merged_poly = join_polys(expanded_polygon)

    # intersect merged_poly and outer_bound
    pc = pyclipper.Pyclipper()
    outer_bound = pyclipper.scale_to_clipper(outer_bound, scale_factor)
    merged_poly = pyclipper.scale_to_clipper(merged_poly, scale_factor)
    pc.AddPaths(outer_bound, pyclipper.PT_CLIP, True)
    pc.AddPaths(merged_poly, pyclipper.PT_SUBJECT, True)
    intersection = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    # if the intersection is precisecly the outer bound, outer bound is cointained in merged_poly
    merged_poly = pyclipper.scale_from_clipper(merged_poly, scale_factor)
    if intersection == outer_bound:
        return (True, merged_poly)
    else:
        return (False, merged_poly)


def error_bound_known_example(order, outer_bound):
    # the known example is a(t) = t^{-k} + t (the limit set will be star shaped)

    poly_points = []
    for poly in outer_bound:
        poly_points.extend(poly)
    hi = max((dist2((0, 0), p) for p in poly_points))**0.5
    lo = 0

    while (hi - lo) > 10**-6:
        # print("lo hi:", lo, hi)
        mid = (hi + lo)/2
        works, last_expanded = bound_contained_known_example(order, outer_bound, mid)
        if works:
            hi = mid
        else:
            lo = mid
    
    return hi, last_expanded

def compute_approx_error(a, r, s, nbr_sample_points):
    sample_points = np.linspace(0, 2*np.pi, 1000)
    rho_lo, rho_hi = find_rho_limits(a, r, s, 1000, 1)
    rhos = np.linspace(rho_lo, rho_hi, 1000)
    C = 0
    for rho in rhos:
        zds = []
        for sample_point in sample_points:
            z = rho*np.exp(1.0j * sample_point)
            zds.append(z)
        zds = np.array(zds)

        second_deriv = np.zeros(len(sample_points), dtype='complex128')
        n = -r
        coeffs = (a.c).astype('complex128')
        while n <= s:
            second_deriv += (n + n*(n-1))* np.power(zds, n) * coeffs[-(n+r+1)]
            n += 1
        C_rho = (np.max(np.abs(np.real(second_deriv))) + np.max(np.abs(np.imag(second_deriv))))
        C = max(C, C_rho)
        # add some margin for error
    return (2*np.pi/(nbr_sample_points))**2 * C * 1.2

def high_order_test():

    gen_new_geom = False
    gen_new_alg = False

    save_path_subset = "my_path"
    save_path_superset = "my_path"

    global nbr_intersections
    nbr_intersections = 0
    r = 400
    s = 1
    # coefficients for largest powers first
    coeff = [0]*(r+s+1)
    coeff[0] = 1.0 + 0.0j
    coeff[27] = 1.0 + 0.0j
    coeff[28] = 1.0 + 0.0j
    coeff[4] = 1.0+0.0j
    coeff[5] = 1.0+0.0j
    coeff[-1] = 1 + 0.0j
    a = np.poly1d(coeff)
    import limit_calc_mpsolve as lc_mps
    if gen_new_alg:
        start = time.time()
        algebraic_limit_points = lc_mps.find_lim_set(a.coef, r, s, 10)
        end = time.time()
        print("Algebraic algorithm took", end - start, "seconds.")
        with open(save_path_subset, "wb") as f:
            pickle.dump(algebraic_limit_points, f)
    else:
        with open(save_path_subset, "rb") as f:
            algebraic_limit_points = pickle.load(f)

    if gen_new_geom:
        start = time.time()
        approx_polygon, approx_superset = find_limit_set(a, r, s, 1000, 1000, nbr_sweeps=2, nbr_area_sweep_points=250, smoothing_window_size=13, outer_bound=True)
        end = time.time()
        print("Geometric algorithm took", end - start, "seconds.")
        with open(save_path_superset, "wb") as f:
            pickle.dump(approx_superset, f)
    else:
        with open(save_path_superset, "rb") as f:
            approx_superset = pickle.load(f)


    fig, ax = plot_polygons(approx_superset, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    
    fig, ax = plot_polygons(approx_superset, "Limit set", check_points_inside=False, poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")

    fig, ax = plot_polygons([], "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    

def sample_compute_error_star():
    global nbr_intersections
    
    gen_new = False

    r = 4
    s = 1
    # coefficients for largest powers first
    coeff = [0]*(r+s+1)
    coeff[0] = 1.0 + 0.0j
    coeff[-1] = 1 + 0.0j
    a = np.poly1d(coeff)
    nbr_angle_sample_points = 2000
    nbr_rhos = (np.rint(np.linspace(10**2, 10**3, 10))).astype(int)
    k = len(nbr_rhos)
    compute_times = []
    hausdorff_dists = []
    nbr_intersections_list = []

    save_folder = "my_save_folder"
    polygon_estimate_error =  compute_approx_error(a, r, s, nbr_angle_sample_points)
    print("Polygon estimate error:", polygon_estimate_error)
    if gen_new:
        for nbr_rho in nbr_rhos:
            nbr_intersections = 0
            start = time.time()
            Lambdab_alg2, Lambdab_bound_alg2 = find_limit_set(a, r, s, nbr_angle_sample_points, nbr_rho, nbr_sweeps=1, nbr_area_sweep_points=nbr_rho//5, smoothing_window_size=13, outer_bound=True)
            end = time.time()
            compute_times.append(end - start)
            print(f"Algorithm for (n, m) = {(nbr_rho, nbr_angle_sample_points)} took", end - start, "seconds")
            nbr_intersections_list.append(nbr_intersections)
            
            hausdorff_dist, expanded_polygon = error_bound_known_example(r+s, Lambdab_bound_alg2)
            hausdorff_dists.append(hausdorff_dist)

            with open(os.path.join(save_folder, "saved_data_one_sweep.pickle"), "wb") as f:
                pickle.dump({"compute_times": compute_times, "hausdorff_dists": hausdorff_dists, "nbr_intersections_list": nbr_intersections_list}, f)
    else:
        assert "saved_data.pickle" in os.listdir(save_folder)
        with open(os.path.join(save_folder, "saved_data_one_sweep.pickle"), "rb") as f:
            stored = pickle.load(f)
            compute_times = stored["compute_times"]
            hausdorff_dists = stored["hausdorff_dists"]
            nbr_intersections_list = stored["nbr_intersections_list"]



    fig = plt.figure()
    sub_axes = []
    sub_axes.append(fig.subplots())
    fig2 = plt.figure()
    sub_axes.append(fig2.subplots())

    sub_axes[0].loglog(compute_times, hausdorff_dists, '--bo', label="measured distance")
    sub_axes[0].set_xlabel("Compute time")
    sub_axes[0].set_ylabel("$d_H(\Lambda(b), \Lambda^{sup})$")
    sub_axes[0].set_title("Convergence rate")
    sub_axes[0].grid()
    A = np.ones((5, 2))
    A[:, 0] = np.log(np.array(compute_times[-5:]))
    Y = np.log(np.array(hausdorff_dists[-5:]))
    fitted_time = np.linalg.lstsq(A, Y, rcond=None)
    print(fitted_time)
    fitted_distances = np.exp(fitted_time[0][1]) * np.power(np.array(compute_times), fitted_time[0][0])
    sub_axes[0].loglog(compute_times, fitted_distances, 'r', label=f"d = $Ct^{{{np.round(fitted_time[0][0], 3)}}}$")
    sub_axes[0].legend()


    sub_axes[1].loglog(nbr_rhos, hausdorff_dists, '--bo', label="measured distance")
    sub_axes[1].set_xlabel("$n$")
    sub_axes[1].set_ylabel("$d_H(\Lambda(b), \Lambda^{sup})$")
    sub_axes[1].set_title("Convergence rate")
    sub_axes[1].grid()
    A = np.ones((k, 2))
    A[:, 0] = np.log(np.array(nbr_rhos[:]))
    Y = np.log(np.array(hausdorff_dists[:]))
    fitted_time = np.linalg.lstsq(A, Y, rcond=None)
    print(fitted_time)
    fitted_distances = np.exp(fitted_time[0][1]) * np.power(np.array(nbr_rhos), fitted_time[0][0])
    sub_axes[1].loglog(nbr_rhos, fitted_distances, 'r', label=f"d = $Cn^{{{np.round(fitted_time[0][0], 3)}}}$")
    sub_axes[1].legend()


def sample_compute_error_one_sweep():
    global nbr_intersections
    nbr_intersections = 0
    r = 1
    s = 3
    # coefficients for largest powers first
    a = np.poly1d([1, -3*(1+1j), 7j, 4*(1-1j), -2])

    start = time.time()
    Lambdab_alg2, Lambdab_bound_alg2 = find_limit_set(a, r, s, 1000, 1250, nbr_sweeps=1, nbr_area_sweep_points=250, smoothing_window_size=13, outer_bound=True)
    end = time.time()
    print("Algorithm 2 took", end - start, "seconds")
    start = time.time()
    algebraic_limit_points = lc.find_lim_set(a.coef, r, s, 2000)
    end = time.time()
    print("Algebraic algorithm took", end - start, "seconds")
    
    x = algebraic_limit_points.real
    y = algebraic_limit_points.imag
    inner_points = list(zip(x, y))
    print("Error bound:", error_bound(inner_points, Lambdab_bound_alg2))

    plot_polygons(Lambdab_bound_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")

    print("Number intersections used", nbr_intersections)

def sample_compute_error_two_sweeps():
    global nbr_intersections
    nbr_intersections = 0
    r = 1
    s = 3
    # coefficients for largest powers first
    a = np.poly1d([1, -3*(1+1j), 7j, 4*(1-1j), -2])

    start = time.time()
    Lambdab_alg2, Lambdab_bound_alg2 = find_limit_set(a, r, s, 1000, 1000, nbr_sweeps=2, nbr_area_sweep_points=250, smoothing_window_size=13, outer_bound=True)
    end = time.time()
    print("Algorithm 2 took", end - start, "seconds")
    start = time.time()
    algebraic_limit_points = lc.find_lim_set(a.coef, r, s, 2000)
    end = time.time()
    print("Algebraic algorithm took", end - start, "seconds")
    
    x = algebraic_limit_points.real
    y = algebraic_limit_points.imag
    inner_points = list(zip(x, y))
    print("Error bound:", error_bound(inner_points, Lambdab_bound_alg2))

    plot_polygons(Lambdab_bound_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")

    print("Number intersections used", nbr_intersections)

def sample_runs_article():
    global nbr_intersections
    nbr_intersections = 0
    r = 1
    s = 3
    # coefficients for largest powers first
    a = np.poly1d([1, -3*(1+1j), 7j, 4*(1-1j), -2])
    start = time.time()
    Lambdab_alg1, Lambdab_bound_alg1 = find_limit_set(a, r, s, 1000, 1250, nbr_sweeps=1, nbr_area_sweep_points=250, smoothing_window_size=13, outer_bound=True)
    end = time.time()
    print("Algorithm 1 took", end - start, "seconds")
    start = time.time()
    Lambdab_alg2, Lambdab_bound_alg2 = find_limit_set(a, r, s, 1000, 1000, nbr_sweeps=2, nbr_area_sweep_points=250, smoothing_window_size=13, outer_bound=True)
    end = time.time()
    print("Algorithm 2 took", end - start, "seconds")
    start = time.time()
    algebraic_limit_points = lc.find_lim_set(a.coef, r, s, 20000)
    end = time.time()
    print("Algebraic algorithm took", end - start, "seconds")
  
    save_folder = "my_save_folder"
    file_ending = ".jpeg"
    plot_polygons(Lambdab_bound_alg1, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg1_bound_zoom{file_ending}", xlim=[-0.15, 0.15], ylim=[-0.15, 0.15], poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    plot_polygons(Lambdab_bound_alg1, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg1_bound{file_ending}", poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    plot_polygons(Lambdab_alg1, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg1_est_zoom{file_ending}", xlim=[-0.15, 0.15], ylim=[-0.15, 0.15], poly_color="navy", poly_opacity=0.6) 
    plot_polygons(Lambdab_alg1, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg1_est{file_ending}", poly_color="navy", poly_opacity=0.6) 

    plot_polygons(Lambdab_bound_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg2_bound_zoom{file_ending}", xlim=[-0.15, 0.15], ylim=[-0.15, 0.15], poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    plot_polygons(Lambdab_bound_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg2_bound{file_ending}", poly_color="navy", poly_opacity=0.6, reverse_poly_color = "red")
    plot_polygons(Lambdab_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg2_est_zoom{file_ending}", xlim=[-0.15, 0.15], ylim=[-0.15, 0.15], poly_color="navy", poly_opacity=0.6) 
    plot_polygons(Lambdab_alg2, "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/alg2_est{file_ending}", poly_color="navy", poly_opacity=0.6) 

    plot_polygons([], "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/algebraic_alg_zoom{file_ending}", xlim=[-0.15, 0.15], ylim=[-0.15, 0.15], poly_color="navy", poly_opacity=0.6) 
    plot_polygons([], "Limit set", complex_points=algebraic_limit_points, check_points_inside=False, save_name=f"{save_folder}/algebraic_alg{file_ending}", poly_color="navy", poly_opacity=0.6) 

    print("Number intersections used", nbr_intersections)
    

if __name__ == "__main__":
    sample_runs_article()

    plt.show()
