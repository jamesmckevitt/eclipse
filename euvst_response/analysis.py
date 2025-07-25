"""
Result analysis functions for M-ECLIPSES instrument response simulations.

This module provides functions for loading, analyzing, and visualizing
instrument response simulation results.
"""

import dill
import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.map
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .fitting import velocity_from_fit, width_from_fit


def load_instrument_response_results(filepath: str | Path) -> Dict[str, Any]:
    """
    Load instrument response results from the new multi-parameter format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the pickled results file.
        
    Returns
    -------
    dict
        Dictionary containing all results and metadata.
    """
    with open(filepath, "rb") as f:
        data = dill.load(f)
    return data


def get_parameter_combinations(results: Dict[str, Any]) -> List[Tuple]:
    """
    Get all parameter combinations that were simulated.
    
    Parameters
    ----------
    results : dict
        Results dictionary from load_instrument_response_results.
        
    Returns
    -------
    list of tuples
        List of parameter combination keys.
    """
    return list(results["results"]["all_combinations"].keys())


def analyse_fit_statistics(
    combination_results: Dict[str, Any],
    rest_wavelength: u.Quantity,
    data_type: str = "dn"
) -> Dict[str, Any]:
    """
    Analyze fit statistics to compute velocity and line width statistics.
    
    Parameters
    ----------
    combination_results : dict
        Results for a specific parameter combination.
    rest_wavelength : u.Quantity
        Rest wavelength for velocity conversion.
    data_type : str, optional
        Either "dn" or "photon" to specify which fit statistics to analyze.
        
    Returns
    -------
    dict
        Dictionary containing velocity and width statistics.
    """
    # Get fit statistics
    fit_stats_key = f"{data_type}_fit_stats"
    if fit_stats_key not in combination_results:
        raise ValueError(f"No {fit_stats_key} found in combination results")
    
    fit_stats = combination_results[fit_stats_key]
    fit_truth = combination_results["ground_truth"]["fit_truth"]
    
    # Calculate velocity statistics from mean and std of fit centers (parameter 1)
    center_mean = fit_stats["mean"][..., 1]  # (nx, ny)
    center_std = fit_stats["std"][..., 1]    # (nx, ny)
    
    # Convert centers to velocities using simple formula
    # v = (lambda - lambda0) / lambda0 * c
    def centers_to_velocity(centers, lambda0):
        """Convert wavelength centers to velocities"""
        # Extract values and units from Quantity objects
        center_values = np.array([q.to_value(lambda0.unit) for q in centers.flat]).reshape(centers.shape)
        centers_quantity = center_values * lambda0.unit
        
        velocity = ((centers_quantity - lambda0) / lambda0 * const.c).to(u.km / u.s)
        return velocity
    
    # Convert to velocities
    v_mean = centers_to_velocity(center_mean, rest_wavelength)
    v_true = centers_to_velocity(fit_truth[..., 1], rest_wavelength)
    v_err = v_true - v_mean
    
    # Convert center std to velocity std using differential: dv/dλ = c/λ
    c = const.c.to(u.km / u.s)
    center_std_clean = np.asarray([q.to(rest_wavelength.unit).value if hasattr(q, 'unit') else q for q in center_std.flat]).reshape(center_std.shape)
    v_std = c * center_std_clean * rest_wavelength.unit / rest_wavelength
    
    # Calculate width statistics from mean and std of fit widths (parameter 2)
    width_mean = fit_stats["mean"][..., 2]   # (nx, ny)
    width_std = fit_stats["std"][..., 2]     # (nx, ny)
    
    # Apply the unit to the whole array, not each value
    width_unit = width_mean.flat[0].unit
    width_mean_values = np.array([q.to_value(width_unit) for q in width_mean.flat]).reshape(width_mean.shape)
    w_mean = width_mean_values * width_unit

    width_std_unit = width_std.flat[0].unit
    width_std_values = np.array([q.to_value(width_std_unit) for q in width_std.flat]).reshape(width_std.shape)
    w_std = width_std_values * width_std_unit
    
    return {
        "v_mean": v_mean,
        "v_std": v_std,
        "v_err": v_err,
        "v_true": v_true,
        "w_mean": w_mean,
        "w_std": w_std,
        "fit_stats": fit_stats,  # Include original fit stats for reference
    }


def get_results_for_combination(
    results: Dict[str, Any], 
    slit_width: u.Quantity = None,
    oxide_thickness: u.Quantity = None, 
    c_thickness: u.Quantity = None,
    aluminium_thickness: u.Quantity = None,
    ccd_temperature: u.Quantity = None,
    vis_sl: u.Quantity = None,
    exposure: u.Quantity = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Get results for a specific parameter combination.
    
    Parameters
    ----------
    results : dict
        Results dictionary from load_instrument_response_results.
    slit_width : u.Quantity, optional
        Slit width with units (e.g., 0.2 * u.arcsec). If None, uses first available.
    oxide_thickness : u.Quantity, optional
        Oxide thickness with units (e.g., 95 * u.nm). If None, uses first available.
    c_thickness : u.Quantity, optional
        Carbon thickness with units (e.g., 20 * u.nm). If None, uses first available.
    aluminium_thickness : u.Quantity, optional
        Aluminium thickness with units (e.g., 1485 * u.AA). If None, uses first available.
    ccd_temperature : u.Quantity, optional
        CCD temperature with units (e.g., -40.0 * u.deg_C). If None, uses first available.
    vis_sl : u.Quantity, optional
        Stray light level with units (e.g., 0 * u.photon / (u.s * u.pixel)). If None, uses first available.
    exposure : u.Quantity, optional
        Exposure time with units (e.g., 80 * u.s). If None, uses first available.
    debug : bool, optional
        If True, print debugging information about available keys.
        
    Returns
    -------
    dict
        Results for the specified parameter combination.
    """
    all_combinations = results["results"]["all_combinations"]
    param_ranges = results["results"]["parameter_ranges"]
    
    if debug:
        print(f"Available param_ranges keys: {list(param_ranges.keys())}")
        print(f"Sample combination keys: {list(all_combinations.keys())[:3]}")
        print(f"Key lengths: {[len(k) for k in list(all_combinations.keys())[:3]]}")
        print(f"Available slit widths: {[sw.to_value(u.arcsec) for sw in param_ranges['slit_widths']]}")
        print(f"Available oxide thicknesses: {[ot.to_value(u.nm) for ot in param_ranges['oxide_thicknesses']]}")
        print(f"Available carbon thicknesses: {[ct.to_value(u.nm) for ct in param_ranges['c_thicknesses']]}")
        print(f"Available aluminium thicknesses: {[at.to_value(u.AA) for at in param_ranges['aluminium_thicknesses']]}")
        print(f"Available CCD temperatures: {param_ranges['ccd_temperatures']} °C")
        print(f"Available stray light values: {[vs.to_value() if hasattr(vs, 'to_value') else vs for vs in param_ranges['vis_sl_vals']]}")
        print(f"Available exposures: {[ex.to_value(u.s) for ex in param_ranges['exposures']]}")
    
    # Check if no parameters were specified at all
    no_params_specified = all(param is None for param in [slit_width, oxide_thickness, c_thickness, 
                                                          aluminium_thickness, ccd_temperature, vis_sl, exposure])
    
    if no_params_specified and len(all_combinations) > 1:
        print(f"Error: No parameters specified, but {len(all_combinations)} combinations are available!")
        print(f"Please specify at least one parameter to select a unique combination.")
        print(f"Use summary_table(results) to see all available parameter combinations.")
        raise ValueError("No parameters specified but multiple combinations exist. Please specify parameters to select a unique combination.")
    
    # Store original None values before filling defaults
    original_params = {
        'slit_width': slit_width,
        'oxide_thickness': oxide_thickness,
        'c_thickness': c_thickness,
        'aluminium_thickness': aluminium_thickness,
        'ccd_temperature': ccd_temperature,
        'vis_sl': vis_sl,
        'exposure': exposure
    }
    
    # FIRST: Check if the specified parameters match multiple combinations
    # before filling in any defaults
    matching_combinations = []
    
    for key in all_combinations.keys():
        key_slit, key_oxide, key_carbon, key_aluminium, key_ccd, key_vis_sl, key_exposure = key
        
        # Check if this combination matches all specified (non-None) parameters
        matches = True
        if slit_width is not None and abs(key_slit - slit_width.to_value(u.arcsec)) > 1e-10:
            matches = False
        if oxide_thickness is not None and abs(key_oxide - oxide_thickness.to_value(u.nm)) > 1e-10:
            matches = False
        if c_thickness is not None and abs(key_carbon - c_thickness.to_value(u.nm)) > 1e-10:
            matches = False
        if aluminium_thickness is not None and abs(key_aluminium - aluminium_thickness.to_value(u.AA)) > 1e-10:
            matches = False
        if ccd_temperature is not None and abs(key_ccd - ccd_temperature.to_value(u.deg_C, equivalencies=u.temperature())) > 1e-10:
            matches = False
        if vis_sl is not None:
            vis_sl_val = vis_sl.to_value() if hasattr(vis_sl, 'to_value') else vis_sl
            if abs(key_vis_sl - vis_sl_val) > 1e-10:
                matches = False
        if exposure is not None and abs(key_exposure - exposure.to_value(u.s)) > 1e-10:
            matches = False
            
        if matches:
            matching_combinations.append(key)
    
    if len(matching_combinations) > 1:
        print(f"Error: Your parameters match {len(matching_combinations)} different combinations!")
        print(f"Please be more specific to select only one combination.")
        print(f"Use summary_table(results) to see all available parameter combinations.")
        print(f"Matching combinations found:")
        for i, combo in enumerate(matching_combinations[:5]):  # Show first 5
            slit, oxide, carbon, aluminium, ccd, vis_sl, exp = combo
            print(f"  {i+1}: slit={slit:.2f}arcsec, oxide={oxide:.1f}nm, carbon={carbon:.1f}nm, "
                  f"Al={aluminium:.0f}Å, CCD={ccd:.1f}°C, stray={vis_sl:.2g}, exp={exp:.1f}s")
        if len(matching_combinations) > 5:
            print(f"  ... and {len(matching_combinations) - 5} more")
        raise ValueError(f"Multiple combinations match your parameters. Please specify more parameters to select a unique combination.")
    elif len(matching_combinations) == 1:
        return all_combinations[matching_combinations[0]]
    
    # If we get here, either no matches or need to use defaults and try exact match
    # Use defaults if not specified, keeping units throughout
    if slit_width is None:
        slit_width = param_ranges["slit_widths"][0]
    if oxide_thickness is None:
        oxide_thickness = param_ranges["oxide_thicknesses"][0]
    if c_thickness is None:
        c_thickness = param_ranges["c_thicknesses"][0]
    if aluminium_thickness is None:
        aluminium_thickness = param_ranges["aluminium_thicknesses"][0]
    if ccd_temperature is None:
        ccd_temperature = param_ranges["ccd_temperatures"][0]  # This should already have units
    if vis_sl is None:
        vis_sl = param_ranges["vis_sl_vals"][0]
    if exposure is None:
        exposure = param_ranges["exposures"][0]
    
    # Convert units to the same format as stored in keys (without units)
    slit_width_val = slit_width.to_value(u.arcsec)
    oxide_thickness_val = oxide_thickness.to_value(u.nm)
    c_thickness_val = c_thickness.to_value(u.nm)
    aluminium_thickness_val = aluminium_thickness.to_value(u.AA)
    ccd_temperature_val = ccd_temperature.to_value(u.deg_C, equivalencies=u.temperature())  # Convert to Celsius value
    vis_sl_val = vis_sl.to_value() if hasattr(vis_sl, 'to_value') else vis_sl
    exposure_val = exposure.to_value(u.s)
    
    # Find matching combination (7-element key format)
    target_key = (slit_width_val, oxide_thickness_val, c_thickness_val, 
                  aluminium_thickness_val, ccd_temperature_val, vis_sl_val, exposure_val)
    
    if debug:
        print(f"Target key: {target_key}")
    
    # Check for exact matches - if multiple exist, warn user
    exact_matches = [key for key in all_combinations.keys() if key == target_key]
    
    if len(exact_matches) > 1:
        print(f"Warning: Found {len(exact_matches)} exact matches for the same parameter combination!")
        print(f"This suggests duplicate entries in the results. Using the first match.")
        return all_combinations[exact_matches[0]]
    elif len(exact_matches) == 1:
        return all_combinations[exact_matches[0]]
    
    # If no exact match, find closest
    closest_key = None
    min_distance = float('inf')
    
    for key in all_combinations.keys():
        distance = sum((a - b)**2 for a, b in zip(key, target_key))
        if distance < min_distance:
            min_distance = distance
            closest_key = key
    
    if closest_key is not None:
        print(f"No exact match found. Using closest combination: {closest_key}")
        print(f"Target was: {target_key}")
        return all_combinations[closest_key]
    else:
        raise ValueError("No matching parameter combination found")


def summary_table(results: Dict[str, Any]) -> None:
    """
    Print a summary table of all parameter combinations and their results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from load_instrument_response_results.
    """
    all_combinations = results["results"]["all_combinations"]
    param_ranges = results["results"]["parameter_ranges"]
    
    print("Parameter Combination Summary")
    print("=" * 120)
    print(f"{'Slit (arcsec)':<12} {'Oxide (nm)':<12} {'Carbon (nm)':<12} {'Al (Å)':<10} {'CCD (°C)':<10} {'Stray Light':<12} {'Exp (s)':<10}")
    print("-" * 120)
    
    for key, combo_results in all_combinations.items():
        slit, oxide, carbon, aluminium, ccd_temp, vis_sl, exposure = key
        params = combo_results["parameters"]
        
        print(f"{slit:<12.2f} {oxide:<12.1f} {carbon:<12.1f} {aluminium:<10.0f} {ccd_temp:<10.1f} {vis_sl:<12.2g} {exposure:<10.1f}")
    
    print("-" * 120)
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Exposure times: {[exp.to_value(u.s) for exp in param_ranges['exposures']]}")


def create_sunpy_maps_from_combo(
    combination_results: Dict[str, Any],
    cube_reb,
    rest_wavelength: u.Quantity = 195.119 * u.AA,
    data_type: str = "dn"
) -> Dict[str, Any]:
    """
    Create SunPy maps from combination results using the new fit statistics structure.
    
    Parameters
    ----------
    combination_results : dict
        Results for a specific parameter combination from get_results_for_combination().
    cube_reb : NDCube
        NDCube with helioprojective WCS to use for all maps.
    rest_wavelength : u.Quantity, optional
        Rest wavelength for velocity conversion (default: 195.119 Å for Fe XII).
    data_type : str, optional
        Either "dn" or "photon" to specify which fit statistics to use for velocity/width maps.
        
    Returns
    -------
    dict
        Dictionary of SunPy maps with keys:
        - 'total_photons': Total photons (summed along wavelength) from first MC iteration
        - 'total_dn': Total DN (summed along wavelength) from first MC iteration
        - 'velocity_from_fit': Velocity from first fit of first MC iteration
        - 'velocity_mean': Mean velocity across all MC iterations
        - 'velocity_std': Velocity uncertainty (standard deviation)
        - 'velocity_err': Velocity error (truth - mean)
        - 'line_width_from_fit': Line width from first fit of first MC iteration  
        - 'line_width_mean': Mean line width across all MC iterations
        - 'line_width_std': Line width uncertainty (standard deviation)
    """
    
    # Extract 2D helioprojective WCS from the cube
    wcs_2d = cube_reb.wcs.celestial.swapaxes(0, 1)
    
    # Get the data arrays - now only first iteration is saved
    first_photon_signal = combination_results["first_photon_signal"]  # Shape: (nx, ny, nwave)
    first_dn_signal = combination_results["first_dn_signal"]         # Shape: (nx, ny, nwave)
    fit_stats_key = f"{data_type}_fit_stats"
    fit_stats = combination_results[fit_stats_key]          # Contains first_fit, mean, std
    
    maps = {}

    # --- Total photons map (before detector effects) ---
    total_photons_data = first_photon_signal.data.sum(axis=2)  # Sum along wavelength
    total_photons_unit = first_photon_signal.unit * u.pix

    maps['total_photons'] = sunpy.map.Map(total_photons_data.T, wcs_2d)
    # 25/07/2025 SunPy failing to pass "unit" keyword to give the map units, so performing manually throughout this function.
    maps['total_photons'].meta['bunit'] = str(total_photons_unit)

    # --- Total DN map (after detector effects) ---
    total_dn_data = first_dn_signal.data.sum(axis=2)  # Sum along wavelength
    total_dn_unit = first_dn_signal.unit * u.pix

    maps['total_dn'] = sunpy.map.Map(total_dn_data.T, wcs_2d)
    maps['total_dn'].meta['bunit'] = str(total_dn_unit)
    
    # --- Get velocity and width analysis for this combination ---
    analysis = analyse_fit_statistics(combination_results, rest_wavelength, data_type)

    # --- Velocity maps ---
    # Velocity from first fit (parameter 1 = center)
    first_fit = fit_stats["first_fit"]  # Shape: (nx, ny, 4)
    center_first = first_fit[..., 1]    # Extract center parameter

    def centers_to_velocity(centers, lambda0):
        """Convert wavelength centers to velocities"""
        # take the units out of the array and apply to the whole array
        center_values = np.array([q.to_value(lambda0.unit) for q in centers.flat]).reshape(centers.shape)
        centers_quantity = center_values * lambda0.unit
        
        velocity = ((centers_quantity - lambda0) / lambda0 * const.c).to(u.km / u.s)
        return velocity

    v_first = centers_to_velocity(center_first, rest_wavelength)

    maps['velocity_from_fit'] = sunpy.map.Map(v_first.value.T, wcs_2d)
    maps['velocity_from_fit'].meta['bunit'] = str(v_first.unit)
    
    maps['velocity_mean'] = sunpy.map.Map(analysis["v_mean"].value.T, wcs_2d)
    maps['velocity_mean'].meta['bunit'] = str(analysis["v_mean"].unit)

    maps['velocity_std'] = sunpy.map.Map(analysis["v_std"].value.T, wcs_2d)
    maps['velocity_std'].meta['bunit'] = str(analysis["v_std"].unit)
    
    # Velocity error (truth - mean)
    maps['velocity_err'] = sunpy.map.Map(analysis["v_err"].value.T, wcs_2d)
    maps['velocity_err'].meta['bunit'] = str(analysis["v_err"].unit)

    # --- Line width maps ---
    # Line width from first fit (parameter 2 = width)
    width_first = first_fit[..., 2]     # Extract width parameter

    # Take units out of the array and apply to the whole array
    values = np.array([q.value for q in width_first.flat]).reshape(width_first.shape)
    width_quantity = values * width_first.flat[0].unit
    # Convert to Angstroms and extract value for SunPy Map
    width_data_clean = width_quantity.to(u.AA).value
    
    maps['line_width_from_fit'] = sunpy.map.Map(width_data_clean.T, wcs_2d)
    
    # Mean line width across all iterations
    # Handle line width data properly
    w_mean = analysis["w_mean"]
    w_mean_data_clean = w_mean.to(u.AA).value

    maps['line_width_mean'] = sunpy.map.Map(w_mean_data_clean.T, wcs_2d)

    # Line width standard deviation (uncertainty)
    w_std = analysis["w_std"]
    w_std_data_clean = w_std.to(u.AA).value
    maps['line_width_std'] = sunpy.map.Map(w_std_data_clean.T, wcs_2d)
    
    # Set appropriate visualization settings for common map types
    # Also ensure correct aspect ratio for all maps
    map_names = list(maps.keys())

    # Set aspect ratio metadata for all maps to ensure correct plotting
    cdelt_x = wcs_2d.wcs.cdelt[0]
    cdelt_y = wcs_2d.wcs.cdelt[1]
    aspect_ratio = cdelt_y / cdelt_x
    for map_name in map_names:
        maps[map_name].plot_settings.update({
            'aspect': aspect_ratio,
        })
    
    # Set specific color maps and ranges
    maps['total_photons'].plot_settings.update(dict(cmap="afmhot", norm="log"))
    maps['total_dn'].plot_settings.update(dict(cmap="afmhot", norm="log"))
    maps['velocity_from_fit'].plot_settings.update(dict(cmap="RdBu_r", vmin=-15, vmax=15))
    maps['velocity_mean'].plot_settings.update(dict(cmap="RdBu_r", vmin=-15, vmax=15))
    maps['velocity_std'].plot_settings.update(dict(cmap="magma", vmin=0))
    maps['line_width_from_fit'].plot_settings.update(dict(cmap="Purples"))
    maps['line_width_mean'].plot_settings.update(dict(cmap="Purples"))
    maps['line_width_std'].plot_settings.update(dict(cmap="Purples"))

    return maps
