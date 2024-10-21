#!/usr/bin/env python3
# Standard library imports
# print the name of the environment used
import getpass
import importlib
import io
import math
import os
import pathlib
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
import warnings
from functools import partial, wraps
from itertools import compress
from pathlib import Path

# Third-party imports
import gillespy2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson
from tqdm.auto import tqdm
from typing import List
import trackpy as tp
tp.quiet(suppress=True)
from multiprocessing import Pool
import multiprocessing
number_cpus = multiprocessing.cpu_count()
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import imageio
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter, binary_dilation, label
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


# Suppress warnings and reset matplotlib parameters
warnings.filterwarnings("ignore")
mpl.rcParams.update(mpl.rcParamsDefault)

# current directory
current_dir = pathlib.Path().absolute()


# Importing libraries
def create_output_folder(folder_name):
    folder_outputs = current_dir.joinpath(folder_name)
    if os.path.exists(folder_outputs):
        shutil.rmtree(folder_outputs)
    folder_outputs.mkdir(parents=True, exist_ok=True)
    return folder_outputs


figSize=800

plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
})
# Assign colors to each species for plotting
colors = [ '#FBD148', '#6BCB77', '#AA66CC','#FF6B6B', '#4D96FF']
species_colors = {
    'G_off': colors[0],
    'G_on': colors[1],
    'R_n': colors[2],
    'R_c': colors[3],
    'P':  colors[4]
}


#@title Plotting configuration
plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
})
# Assign colors to each species for plotting
colors = [ '#FBD148', '#6BCB77', '#AA66CC','#FF6B6B', '#4D96FF']
species_colors = {
    #'G_off': colors[0],
    #'G_on': colors[1],
    'R_n': colors[2],
    'R_c': colors[3],
    'P':  colors[4]
}


# Set up paths
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
if src_dir is not None:
    sys.path.append(str(src_dir))
    # Import custom modules
    import qbio_paper as qbio
    # Reload custom modules
    importlib.reload(qbio)
else:
    print("Source directory not found. Please check the path to 'src' directory.")



# def plotting_deterministic(time,concentrations_species,species_colors,drug_application_time=None,ylim_val=False,save_figure=True,plot_name='det.jpg'):
#     plt.figure(figsize=(8, 4))
#     for species, color in species_colors.items():
#         plt.plot(time, concentrations_species[species], color=color, label=species,lw=4)
#     if not drug_application_time is None:
#         plt.axvline(x=drug_application_time, color='k', linestyle='--', label= r'$t_{drug}$',lw=1.5)
#     plt.xlabel('Time')
#     plt.ylabel('Concentration')
#     plt.title('Deterministic Dynamics')
#     plt.legend(loc='upper right')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     if ylim_val:
#         plt.ylim(0,ylim_val)
#     if save_figure == True: 
#         plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)  
#     plt.show()


def plotting_deterministic(time, concentrations_species, species_colors, drug_application_time=None, ylim_val=False, save_figure=True, plot_name='det.jpg',folder_outputs=None):
    plt.figure(figsize=(8, 4))
    for species, color in species_colors.items():
        if species in concentrations_species:
            plt.plot(time, concentrations_species[species], color=color, label=species, lw=4)
    if drug_application_time is not None:
        plt.axvline(x=drug_application_time, color='k', linestyle='--', label=r'$t_{drug}$', lw=1.5)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Deterministic Dynamics')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    if ylim_val:
        plt.ylim(0, ylim_val)
    if save_figure:
        if folder_outputs is None:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
        else:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)  
    plt.show()

def plotting_stochastic(time, trajectories_species,species_colors,drug_application_time=None,ylim_val=False,save_figure=True,plot_name='ssa.jpg',folder_outputs=None):
    def plot_species_trajectories(time, trajectories_species, species_name, color):
        # Extract the trajectories for the species
        trajectories = trajectories_species[species_name]
        # Calculate the mean and standard deviation across all trajectories
        mean_trajectories = np.mean(trajectories, axis=0)
        std_trajectories = np.std(trajectories, axis=0)
        # Plot mean concentration with standard deviation as shaded area
        plt.plot(time, mean_trajectories, '-', color=color, label=species_name, lw=4)
        plt.fill_between(time, mean_trajectories - std_trajectories, mean_trajectories + std_trajectories, color=color, alpha=0.1)
    plt.figure(figsize=(8, 4))
    # Plot each species
    for species, color in species_colors.items():
        plot_species_trajectories(time, trajectories_species, species, color)
    # Mark the drug application time
    if not drug_application_time is None:
        plt.axvline(x=drug_application_time, color='k', linestyle='--', label=r'$t_{drug}$', lw=1.5)
    # Set plot details
    plt.xlabel('Time')
    plt.ylabel('Number of Molecules')
    plt.title('Stochastic Dynamics')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    if ylim_val:
        plt.ylim(0,ylim_val)
    if save_figure == True: 
        if folder_outputs is None:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
        else:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()


def plotting_stochastic_dist(time, trajectories_species, species_name, species_color_dist, time_points, ylim_val=False,save_figure=True,plot_name='ssa_dist.jpg',folder_outputs=None):
    def plot_species_trajectories(ax, time, trajectories_species, species_name, color, time_point):
        # Extract the trajectories for the species at the specific time point
        trajectories = trajectories_species[species_name][:, time_point]
        # Plot distribution of species count at the specified time point
        ax.hist(trajectories, bins=20, color=color, alpha=0.7, label=f"{species_name} at time={time[time_point]}")
        ax.legend(loc='upper right')
    # Set up a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()  # Flatten the 2x2 array to easily loop over it
    # Plot the species distribution at each of the four user-selected time points
    for i, time_point in enumerate(time_points):
        plot_species_trajectories(axs[i], time, trajectories_species, species_name, species_color_dist, time_point)
        axs[i].set_xlabel('Number of Molecules')
        axs[i].set_ylabel('Frequency')
        if ylim_val:
            axs[i].set_ylim(0, ylim_val)
    # Set a general title
    plt.suptitle(f'Distribution of {species_name} at Different Time Points')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplots to fit into the figure area
    if save_figure == True: 
        if folder_outputs is None:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
        else:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()


def plotting_stochastic_dist_all_species(time, trajectories_species, species_colors, time_point, ylim_val=False, xlim_vals=None, save_figure=True, plot_name='ssa_dist_all_species.jpg',folder_outputs=None):
    n_species = len(species_colors)  # Number of species to plot
    n_cols = n_species  # Number of columns in subplot grid
    n_rows = 1
    def plot_species_trajectories(ax, time, trajectories_species, species_name, color, time_point, xlim_val=None):
        def bins_histogram(data):
            if data.max() > 60:
                step_bins =2
            else:
                step_bins=1
            bins = np.arange(np.floor(data.min()), np.ceil(data.max()), step_bins)
            return bins
        # Extract the trajectories for the species at the specific time point
        trajectories = trajectories_species[species_name][:, time_point]
        # Plot distribution of species count at the specified time point
        bins = bins_histogram(trajectories)
        # Plot distribution of species count at the specified time point using defined bins
        ax.hist(trajectories, bins=bins, color=color, alpha=0.7, label=f"{species_name}")  # Label includes species name
        #ax.hist(trajectories, bins=30, color=color, alpha=0.7, label=f"{species_name}") # at time={time[time_point]}")
        ax.legend(loc='upper right')
        ax.set_xlabel('Number of Molecules')
        ax.set_ylabel('Frequency')
        if ylim_val:
            ax.set_ylim(0, ylim_val)
        if xlim_val:  # Set the x-axis limit if specified
            ax.set_xlim(xlim_val)
    # Set up subplot grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), squeeze=False)
    axs = axs.flatten()  # Flatten the array to easily loop over it
    # Plot the species distribution at the selected time point
    for i, (species_name, species_color) in enumerate(species_colors.items()):
        xlim_val = xlim_vals.get(species_name, None) if xlim_vals else None  # Get the x-axis limit for the species if specified
        plot_species_trajectories(axs[i], time, trajectories_species, species_name, species_color, time_point, xlim_val)
    for ax in axs[i+1:]:  # Hide unused subplots
        ax.set_visible(False)
    # Set a general title
    plt.suptitle(f'Distribution of Species at Time={time[time_point]}',fontsize=20 )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure == True: 
        if folder_outputs is None:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
        else:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()
        

def plotting_combined(time, trajectories_species, species_colors, drug_application_time=None, ylim_val=None, time_point=10, save_figure=True, plot_name='ssa_tc_dist.jpg',folder_outputs=None):
    # Create a single row with four columns
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), squeeze=False)
    axs = axs.flatten()  # Flatten the array for easier indexing
    def plot_time_courses(ax, time, trajectories_species, species_colors, drug_application_time):
        # Assuming trajectories_species is a list of dicts, each dict representing a single "trajectory" 
        # and keys are species names pointing to a 2D array (trajectories, time points)
        for species_name, color in species_colors.items():
            # Aggregate data across all trajectories for the species
            all_trajectories = np.vstack([trajectory[species_name] for trajectory in trajectories_species])
            mean_trajectories = np.mean(all_trajectories, axis=0)
            std_trajectories = np.std(all_trajectories, axis=0)
            ax.plot(time, mean_trajectories, '-', color=color, label=species_name, lw=2)
            ax.fill_between(time, mean_trajectories - std_trajectories, mean_trajectories + std_trajectories, color=color, alpha=0.1)
            if drug_application_time is not None:
                ax.axvline(x=drug_application_time, color='k', linestyle='--', label='Drug application', lw=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Molecules')
            ax.legend(loc='upper right', fontsize=14)
            ax.set_title('Time Courses for All Species')
            if ylim_val:
                ax.set_ylim(0, ylim_val)
    def plot_species_distribution(ax, time, trajectories_species, species_name, color, time_point):
        def bins_histogram(data):
            if data.max() > 60:
                step_bins =2
            else:
                step_bins=1
            bins = np.arange(np.floor(data.min()), np.ceil(data.max()), step_bins)
            return bins
        # Aggregate data across all trajectories for the species at the specified time point
        all_trajectories_at_time_point = np.hstack([trajectory[species_name][ time_point] for trajectory in trajectories_species])
        bins = bins_histogram(all_trajectories_at_time_point)
        ax.hist(all_trajectories_at_time_point, bins=bins, color=color, alpha=0.7, label=f"{species_name}")
        ax.legend(loc='upper right')
        ax.set_xlabel('Number of Molecules')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution at Time={time[time_point]}')
    # Plot time courses in the first subplot
    plot_time_courses(axs[0], time, trajectories_species, species_colors, drug_application_time)
    species_list = list(species_colors.keys())
    # Assume up to three additional species for distribution plots
    for i, species_name in enumerate(species_list[:3]):  # Limit to three species
        plot_species_distribution(axs[i + 1], time, trajectories_species, species_name, species_colors[species_name], time_point)
    # Adjust remaining subplots if fewer than three species are provided
    for j in range(i + 2, 4):
        axs[j].set_visible(False)
    plt.tight_layout()
    if save_figure:
        if folder_outputs is None:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1) 
        else:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()



class GeneExpressionModel(gillespy2.Model):
    def __init__(self, parameter_values, initial_conditions, mode='continuous'):
        super().__init__('GeneExpressionModel')
        # Add parameters from the dictionary to the model
        for name, expression in parameter_values.items():
            self.add_parameter(gillespy2.Parameter(name=name, expression=expression))
        # Define species with initial conditions and add them to the model
        species_list = [
            gillespy2.Species(name='R_n', initial_value=initial_conditions.get('R_n', 0), mode=mode),
            gillespy2.Species(name='R_c', initial_value=initial_conditions.get('R_c', 0), mode=mode),
            gillespy2.Species(name='P', initial_value=initial_conditions.get('P', 0), mode=mode)
        ]
        self.add_species(species_list)
        # Define reactions and add them to the model
        reactions = [
            gillespy2.Reaction(name='mRNA_production', reactants={}, products={species_list[0]: 1}, rate=self.listOfParameters['k_r']),
            gillespy2.Reaction(name='mRNA_transport', reactants={species_list[0]: 1}, products={species_list[1]: 1}, rate=self.listOfParameters['k_t']),
            gillespy2.Reaction(name='protein_production', reactants={species_list[1]: 1}, products={species_list[1]: 1, species_list[2]: 1}, rate=self.listOfParameters['k_p']),
            gillespy2.Reaction(name='nuclear_mRNA_decay', reactants={species_list[0]: 1}, products={}, rate=self.listOfParameters['gamma_r']),
            gillespy2.Reaction(name='cytoplasm_mRNA_decay', reactants={species_list[1]: 1}, products={}, rate=self.listOfParameters['gamma_r']),
            gillespy2.Reaction(name='protein_decay', reactants={species_list[2]: 1}, products={}, rate=self.listOfParameters['gamma_p'])
        ]
        self.add_reaction(reactions)
        # Define the simulation time span
        self.timespan(np.linspace(0, 100, 101))

def initialize_model(parameter_values, initial_conditions, mode='continuous', apply_drug=False, inhibited_parameters=None):
    model = GeneExpressionModel(parameter_values, initial_conditions, mode=mode)
    for species in model.listOfSpecies.values():
        species.mode = mode
    if apply_drug and inhibited_parameters is not None:
        for param, value in inhibited_parameters.items():
            if param in model.listOfParameters:
                model.listOfParameters[param].expression = str(value)
    return model

def run_simulation_phase(model_initializer, parameter_values, initial_conditions, simulation_end, number_of_trajectories, apply_drug=False, inhibited_parameters=None, simulation_type='discrete', burn_in_time=None):
    model = model_initializer(parameter_values, initial_conditions, mode=simulation_type, apply_drug=apply_drug, inhibited_parameters=inhibited_parameters)
    total_timespan = np.linspace(0, simulation_end, num=int(simulation_end) + 1)
    model.timespan(total_timespan)
    if simulation_type == 'discrete':
        results = model.run(solver=gillespy2.TauLeapingSolver, number_of_trajectories=number_of_trajectories)
        list_all_results = []
        for n in range(number_of_trajectories):
            species_trajectories = {species: results[n][species] for species in model.listOfSpecies.keys()}
            if burn_in_time is not None:
                burn_in_index = burn_in_time if burn_in_time < len(total_timespan) else len(total_timespan) - 1
                species_trajectories = {species: species_trajectories[species][burn_in_index:] for species in model.listOfSpecies.keys()}
            list_all_results.append(species_trajectories)
        return list_all_results
    else:
        result = model.run(solver=gillespy2.ODESolver)
        if burn_in_time is not None:
            trajectories_species = {species: result[0][species][burn_in_time:] for species in model.listOfSpecies.keys()}
        else:
            trajectories_species = {species: result[0][species] for species in model.listOfSpecies.keys()}
        return trajectories_species



def simulate_model(parameter_values, initial_conditions, total_simulation_time, simulation_type, burn_in_time=None, drug_application_time=None, inhibited_parameters=None, number_of_trajectories=1):
    """
    Simulate a model either as deterministic or stochastic based on user input,
    including optional burn-in and drug application phases.
    """
    # Determine the end time for the initial simulation phase
    # This phase may include the burn-in period (if any) and extends up to the drug application time or the total simulation time
    if burn_in_time is None or burn_in_time < 50:
        burn_in_time = None
    if burn_in_time is not None:
        end_time_initial_phase = drug_application_time + burn_in_time if drug_application_time is not None else total_simulation_time+burn_in_time
    else:
        end_time_initial_phase = drug_application_time if drug_application_time is not None else total_simulation_time
    # The run_simulation_phase function will only return results after the burn-in period, adjusting the timespan accordingly
    trajectories_initial = run_simulation_phase( initialize_model, parameter_values=parameter_values, initial_conditions=initial_conditions, simulation_end=end_time_initial_phase, number_of_trajectories=number_of_trajectories, apply_drug=False,inhibited_parameters= None, simulation_type=simulation_type, burn_in_time=burn_in_time)  
    # If there's a drug application phase
    if drug_application_time is not None:
        drug_simulation_end = total_simulation_time - drug_application_time
        if simulation_type == 'continuous':
            updated_initial_conditions = {species: trajectories_initial[species][-1] for species in trajectories_initial}
            trajectories_drug = run_simulation_phase( initialize_model, parameter_values=parameter_values, initial_conditions=updated_initial_conditions, simulation_end=drug_simulation_end, number_of_trajectories=1, apply_drug=True, inhibited_parameters=inhibited_parameters, simulation_type=simulation_type)
            trajectories_species = {species: np.concatenate([trajectories_initial[species], trajectories_drug[species][1:]])
                                    for species in trajectories_initial}
        else:
            all_results_after_drug = []
            for i in range(number_of_trajectories):
                updated_initial_conditions = {species: trajectories_initial[i][species][-1] for species in trajectories_initial[i].keys()}
                trajectories_drug = run_simulation_phase( initialize_model, parameter_values=parameter_values, initial_conditions=updated_initial_conditions, simulation_end=drug_simulation_end, number_of_trajectories=1, apply_drug=True, inhibited_parameters=inhibited_parameters, simulation_type=simulation_type)
                all_results_after_drug.append(trajectories_drug[0])
            # append the results from the first phase to the results from the second phase
            trajectories_species = {}
            #if len(trajectories_initial) > 0 and len(all_results_after_drug) > 0:
            for species in trajectories_initial[0].keys():
                species_data_across_trajectories = []
                for i in range(number_of_trajectories):
                    initial_data = trajectories_initial[i][species]
                    after_drug_data = all_results_after_drug[i][species]
                    concatenated_data = np.concatenate([initial_data[:-1], after_drug_data])
                    species_data_across_trajectories.append(concatenated_data)
                trajectories_species[species] = np.stack(species_data_across_trajectories)
    else:
        trajectories_species = trajectories_initial
    # creating a vector for time span
    time = np.linspace(0, total_simulation_time, num=total_simulation_time + 1)
    return time, trajectories_species


def calculate_effective_kt(D, k_diff_r, transport_rate, model_type):
    """
    Calculate the effective rate (k_t) for RNA transport from the nucleus to the cytosol,
    incorporating the effects of diffusion to the nuclear envelope and transport across it,
    adjusted for the dimensionality of the simulation (2D or 3D).

    Parameters:
    - D (float): Distance to the nuclear envelope (assumed to be in the same units as used for k_diff_r).
    - k_diff_r (float): Diffusion rate of RNA to the nuclear envelope.
    - transport_rate (float): Rate of RNA transport from the nucleus to the cytosol.
    - model_type: The dimensionality of the simulation (2D or 3D).

    Returns:
    - k_t (float): Effective rate of RNA transport from nucleus to cytosol.
    """
    # converting diameter to radius
    D = D/2
    # Adjust the diffusion calculation based on dimensionality
    if model_type == '2D':
        # In 2D, diffusion might be more efficient due to less volume to cover
        T_diff = D**2 / (4 * k_diff_r)  # Adjusted for 2D
    elif model_type == '3D':
        # In 3D, diffusion covers more volume
        T_diff = D**2 / (6 * k_diff_r)  # Adjusted for 3D
    else:
        raise ValueError("Dimension must be 2 or 3.")

    # Calculate T_transport as the inverse of transport_rate
    T_transport = 1 / transport_rate
    
    # Calculate the total effective time (T_total)
    T_total = T_diff + T_transport
    
    # Calculate the effective rate (k_t) as the inverse of T_total
    k_t = 1 / T_total
    
    return k_t








def plot_time_courses_all_species(trajectories,drug_application_time=None,max_y_val=None):
    time_steps = trajectories['time_steps']
    # Initialize counts for RNA and Protein, both in cytosol and nucleus
    counts = {
        'RNA': {'cytosol': [0] * len(time_steps), 'nucleus': [0] * len(time_steps)},
        'Protein': {'cytosol': [0] * len(time_steps), 'nucleus': [0] * len(time_steps)}
    }
    # Function to find the index of the closest time step
    def find_closest_time_step_index(target_time):
        return min(range(len(time_steps)), key=lambda i: abs(time_steps[i] - target_time))
    # Process RNA and Protein trajectories
    for entity_type in ['RNA', 'Protein']:
        entity_trajectories = trajectories[f'{entity_type}_trajectories']
        for entity_list in entity_trajectories.values():
            for snapshot in entity_list:
                time_step_index = find_closest_time_step_index(snapshot['time'])
                location = 'cytosol' if snapshot['in_cytosol'] else 'nucleus'
                counts[entity_type][location][time_step_index] += 1
    plt.figure(figsize=(8, 5))
    # Plot RNA in Nucleus
    plt.plot(time_steps, counts['RNA']['nucleus'], label='RNA in Nucleus', linestyle=':', marker='o', color=species_colors['R_n'])
    # Plot RNA in Cytosol
    plt.plot(time_steps, counts['RNA']['cytosol'], label='RNA in Cytosol', linestyle=':', marker='s', color=species_colors['R_c'])
    # plot total RNA
    #plt.plot(time_steps, total_RNA_counts_per_time_step, label='Total RNA', linestyle='-', marker='o', color='y', linewidth=2)
    # Plot total Protein
    #plt.plot(time_steps, total_Protein_counts_per_time_step, label='Total Protein', linestyle='-', marker='o', color=species_colors['P'], linewidth=2)
    # Plot Protein in Nucleus
    #plt.plot(time_steps, counts['Protein']['nucleus'], label='Protein in Nucleus', linestyle='-', marker='o', color='cyan')
    if not drug_application_time is None:
        plt.axvline(x=drug_application_time, color='k', linestyle='--', label= r'$t_{drug}$',lw=1.5)
    
    if max_y_val is not None:
        plt.ylim(0,max_y_val)
    # Plot Protein in Cytosol
    plt.plot(time_steps, counts['Protein']['cytosol'], label='Protein in Cytosol', linestyle='--', marker='s', color=species_colors['P'])
    plt.xlabel('Time')
    plt.ylabel('Molecule Counts')
    plt.title('Molecule Counts Over Time in Nucleus and Cytosol')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_particle_positions( trajectories, simulation_volume_size, masks_nucleus, masks_cytosol, simulation_type='3D', figsize=(12, 12), time_step_index=None, plot_as_trajectory=True, create_gif=False, elev_val=None, azim_val=None, iteration_index=0, output_folder='temp_plots', show_axes=False, show_time_stamp=False,ax=None):

    # Decide if the plot should be 2D or 3D based on the simulation type
    is_3D = simulation_type == '3D'
    fig = plt.figure(figsize=figsize)
    #if is_3D:
    #    ax = fig.add_subplot(111, projection='3d', adjustable='box')
    #else:
    #    ax = fig.add_subplot(111)
        
    # Initialize plot
    created_ax = False
    if ax is None:
        created_ax = True
        fig = plt.figure(figsize=figsize)
        if is_3D:
            ax = fig.add_subplot(111, projection='3d', adjustable='box')
        else:
            ax = fig.add_subplot(111)
            # Set the background color for the figure
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
    # Define colors
    rna_color = species_colors['R_c']
    protein_color = species_colors['P']
    TS_color = 'lime'
    
    def plot_positions(ax, positions, color, label, is_3D, size=10):
        if positions and len(positions) > 0:
            positions_array = np.array(positions)
            if is_3D:
                xs, ys, zs = positions_array.T
                ax.scatter(xs, ys, zs, color=color, s=size, label=label, depthshade=False)
            else:
                xs, ys = positions_array.T[:2]
                ax.scatter(xs, ys, color=color, s=size, label=label)
    def plot_complete_trajectories(ax, trajectory_data, color, is_3D, markersize=1, linewidth=0.5, time_step_index=None):
        for entity_id, trajectory in trajectory_data.items():
            all_positions = np.array([snapshot['position'] for snapshot in trajectory if time_step_index is None or snapshot['time'] <= time_step_index])
            if len(all_positions) > 1:
                if is_3D:
                    xs, ys, zs = all_positions.T
                    ax.plot(xs, ys, zs, color=color, linestyle='-', marker='o', markersize=markersize, linewidth=linewidth, label=f"ID {entity_id}")
                else:
                    xs, ys = all_positions[:, 0], all_positions[:, 1]
                    ax.plot(xs, ys, color=color, linestyle='-', marker='o', markersize=markersize, linewidth=linewidth, label=f"ID {entity_id}")
            elif len(all_positions) == 1:
                if is_3D:
                    ax.scatter(all_positions[0][0], all_positions[0][1], all_positions[0][2], color=color, s=markersize, label=f"ID {entity_id}")
                else:
                    ax.scatter(all_positions[0][0], all_positions[0][1], color=color, s=markersize, label=f"ID {entity_id}")
    def plot_TS(TS_trajectory, color=TS_color, time_step_index=None, markersize=50):
        if time_step_index is not None:
            ts_data = [data for data in TS_trajectory if data['time'] <= time_step_index]
        else:
            ts_data = TS_trajectory
        for ts_info in ts_data:
            position = ts_info['position']
            ts_color = color if ts_info['state'] else 'gray'
            if is_3D:
                ax.scatter(*position, color=ts_color, s=markersize, label='TS', edgecolor='black', zorder=5)
            else:
                ax.scatter(*position[:2], color=ts_color, s=markersize, label='TS', edgecolor='black', zorder=5)
    def plot_surface_or_contour(ax, mask, color, is_3D):
        if is_3D:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=(1, 1, 1), step_size=1)
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=color, lw=0.1, alpha=0.02)
        else:
            contours = measure.find_contours(mask, level=0.5)
            for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=4)
    def get_positions_at_time_step(trajectories, time_step_index=None):
        time_steps = trajectories['time_steps']
        RNA_trajectories = trajectories['RNA_trajectories']
        Protein_trajectories = trajectories['Protein_trajectories']
        def extract_positions_at_index(entity_trajectories, target_index):
            positions = []
            for entity_list in entity_trajectories.values():
                for snapshot in entity_list:
                    if snapshot['time'] == time_steps[target_index]:
                        positions.append(snapshot['position'])
            return positions
        if time_step_index is not None and time_step_index < len(time_steps):
            RNA_positions = extract_positions_at_index(RNA_trajectories, time_step_index)
            Protein_positions = extract_positions_at_index(Protein_trajectories, time_step_index)
        else:
            RNA_positions, Protein_positions = [], []
        return RNA_positions, Protein_positions
    plot_surface_or_contour(ax, masks_nucleus, 'lightcoral', is_3D)
    plot_surface_or_contour(ax, masks_cytosol, 'lightgray', is_3D)
    if plot_as_trajectory and time_step_index is not None:
        plot_complete_trajectories(ax=ax, trajectory_data=trajectories['RNA_trajectories'], color=rna_color, is_3D=is_3D, markersize=1, linewidth=0.5, time_step_index=time_step_index)
        plot_complete_trajectories(ax=ax, trajectory_data=trajectories['Protein_trajectories'], color=protein_color, is_3D=is_3D, markersize=0.5, linewidth=0.5, time_step_index=time_step_index)
        plot_TS(trajectories['TS_trajectory'], TS_color, time_step_index=time_step_index, markersize=60)
    elif time_step_index is not None:
        RNA_positions, Protein_positions = get_positions_at_time_step(trajectories, time_step_index)
        plot_positions(ax, RNA_positions, rna_color, 'RNA', is_3D, 15)
        plot_positions(ax, Protein_positions, protein_color, 'Protein', is_3D, 15)
        plot_TS(trajectories['TS_trajectory'], TS_color, time_step_index=time_step_index, markersize=60)
    else:
        plot_complete_trajectories(ax=ax, trajectory_data=trajectories['RNA_trajectories'], color=rna_color, is_3D=is_3D, markersize=1)
        plot_complete_trajectories(ax=ax, trajectory_data=trajectories['Protein_trajectories'], color=protein_color, is_3D=is_3D, markersize=0.5)
        plot_TS(trajectories['TS_trajectory'], TS_color, None, markersize=60)
    if show_time_stamp and time_step_index is not None:
        ax.set_title(f'Time: {time_step_index}', fontsize=20)
    if is_3D and elev_val is not None and azim_val is not None:
        ax.view_init(elev=elev_val, azim=azim_val)
    if not show_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if is_3D:
            ax.set_zlabel('Z')
    ax.set_xlim([0, simulation_volume_size[0]])
    ax.set_ylim([0, simulation_volume_size[1]])
    if is_3D:
        ax.set_zlim([0, simulation_volume_size[2]])
    if created_ax:
        if create_gif:
            os.makedirs(output_folder, exist_ok=True)
            filepath = os.path.join(output_folder, f'image_{iteration_index:04d}.png')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, transparent=False)
            plt.close(fig)
        else:
            plt.show()


def generate_frame(args):
    trajectories, simulation_volume_size, masks_nucleus, masks_cytosol,simulation_type, figsize, time_step_index, output_folder, index, elevation, azimuth, show_axes,show_time_stamp,plot_as_trajectory= args
    # This function will call your existing plotting function with the appropriate arguments for each frame
    plot_particle_positions( trajectories, simulation_volume_size, masks_nucleus, masks_cytosol,simulation_type, figsize, time_step_index, create_gif=True, elev_val=elevation, azim_val=azimuth, iteration_index=index, output_folder=output_folder, show_axes=show_axes,show_time_stamp=show_time_stamp,plot_as_trajectory=plot_as_trajectory)
    return os.path.join(output_folder, f'image_{index:04d}.png')
def generate_gif_multiprocessing( trajectories, simulation_volume_size, masks_nucleus, masks_cytosol, simulation_type, figsize,time_step_index=None, output_folder='temp_plots', number_steps=18, show_axes=False, rotate_gif=False, time_course_gif=False, max_time=False, gif_filename='animation_cell.gif',plot_as_trajectory=False):
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, filename))
        os.rmdir(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    if simulation_type == '2D' and rotate_gif:
        raise ValueError("Cannot rotate a 2D simulation. Set rotate_gif=False.")
    args_list = []
    if rotate_gif:
        iteration_vector = np.linspace(0, 360, number_steps).astype(int)
        for index, azimuth in enumerate(iteration_vector):
            args = (trajectories, simulation_volume_size, masks_nucleus, masks_cytosol,simulation_type, figsize, time_step_index, output_folder, index, 25, azimuth, show_axes,False,plot_as_trajectory)
            args_list.append(args)
    if time_course_gif:
        iteration_vector = np.linspace(0, max_time, number_steps).astype(int)
        for index, time_point in enumerate(iteration_vector):
            args = (trajectories, simulation_volume_size, masks_nucleus, masks_cytosol, simulation_type, figsize, time_point, output_folder, index, None, None, show_axes,True,plot_as_trajectory)
            args_list.append(args)
    # Use multiprocessing to generate frames
    with Pool() as pool:
        image_files = pool.map(generate_frame, args_list)
    # Compile the selected images into a GIF
    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        for filepath in image_files:
            writer.append_data(imageio.imread(filepath))
    # Optionally, remove the images after creating the GIF
    for filename in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, filename))
    print(f"GIF saved to {gif_filename}")
    return Image(filename=gif_filename)




class GeneExpressionSimulator:
    def __init__(self, params):
        self.TS_state = False  # Initial state of the Transcription Site (TS)
        self.RNAs = {}
        self.Proteins = {}
        self.next_rna_id = 1
        self.next_protein_id = 1
        self.simulation_type = params.get('simulation_type', '3D')  # Default to '3D' if not specified
        dim = 2 if self.simulation_type == '2D' else 3
        simulation_volume_size = params['simulation_volume_size'][:dim]
        center_of_box = np.array(simulation_volume_size) / 2
        self.center_of_box = center_of_box
        self.simulation_volume_size = simulation_volume_size
        self.nucleus_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.cytosol_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.nucleus_size = params['nucleus_size']
        if params['position_TS']=='center':
            self.transcription_site = center_of_box
        else:
            self.transcription_site = self.find_random_TS_position_inside_nucleus()        
        self.RNA_trajectories = {}  # To store RNA positions over time.
        self.Protein_trajectories = {}  # To store Protein positions over time.
        self.frame_rate =  params['frame_rate']    # Save data every 1 second
        self.time_steps = []  # To store time steps at specified frame rate
        self.TS_trajectory = []  # To store TS trajectory with time step and position
        self.transport_rate = params['transport_rate']
        self.small_distance_outside_nucleus = 1 # Small distance outside nucleus for RNA transport
        self.transport_zone_threshold = 2 # Threshold distance for RNA near nuclear envelope, in pixels
        self.simulation_type = params['simulation_type'] # Type of simulation. options: '3D', '2D'
        self.movement_protein_into_nucleus = params['movement_protein_into_nucleus']  # Allow protein movement in nucleus
        if 'burnin_time' in params:
            self.burnin_time = params['burnin_time']# Time to run the simulation without saving data
        else:
            self.burnin_time = 0
        self.next_save_time = self.burnin_time   # Time for next save
        self.total_time = params['total_time']+self.burnin_time  # Total time for the simulation
        self.drug_application_time= params['drug_application_time']+self.burnin_time  # Time to apply drug
        # Creating the rates for the model
        self.rates = params.copy()
        # Apply drug effect if specified    
        self.inhibited_parameters= params['inhibited_parameters']
        self.apply_drug = params['apply_drug']   
        self.parameters_updated = False
        
    def update_rates_with_drug_effect(self):
        for param, value in self.inhibited_parameters.items():
            if param in self.rates:
                self.rates[param] = value
    
    def find_random_TS_position_inside_nucleus(self):
        dim = 2 if self.simulation_type == '2D' else 3
        # Adjust for dimensionality of simulation_volume_size
        volume_size = np.array(self.simulation_volume_size[:dim])
        while True:
            random_pos = np.random.rand(dim) * volume_size  # Ensure volume_size matches dimension
            if self.is_within_nucleus(pos =random_pos.tolist(),nucleus_diameter=np.min(self.nucleus_size)):
                return random_pos
        
    def save_state(self, current_time):
        current_time_int = int(current_time - self.burnin_time)  # Adjust time to start from 0 after burnin
        self.time_steps.append(current_time_int)
        
        for rna_id, rna_info in self.RNAs.items():
            if rna_id not in self.RNA_trajectories:
                self.RNA_trajectories[rna_id] = []
            rna_snapshot = rna_info.copy()
            rna_snapshot['time'] = current_time_int
            rna_snapshot['id'] = rna_id
            rna_snapshot['position'] = self.RNAs[rna_id]['position']
            self.RNA_trajectories[rna_id].append(rna_snapshot)
        
        for protein_id, protein_info in self.Proteins.items():
            if protein_id not in self.Protein_trajectories:
                self.Protein_trajectories[protein_id] = []
            protein_snapshot = protein_info.copy()
            protein_snapshot['time'] = current_time_int
            protein_snapshot['id'] = protein_id
            protein_snapshot['position'] = self.Proteins[protein_id]['position']
            self.Protein_trajectories[protein_id].append(protein_snapshot)
        
        # For the transcription site, if it also has an 'id', include it as well
        TS_info = {
            'position': self.transcription_site.tolist(),
            'state': self.TS_state,
            'time': current_time_int
        }
        self.TS_trajectory.append(TS_info)
    
    def generate_masks(self):
        # Generate meshgrid according to the simulation type
        if self.simulation_type == '2D':
            x, y = np.meshgrid(np.linspace(0, self.simulation_volume_size[0] - 1, self.simulation_volume_size[0]),
                            np.linspace(0, self.simulation_volume_size[1] - 1, self.simulation_volume_size[1]),
                            indexing='ij')
            positions = np.stack((x, y), axis=-1)  # Shape: [dim_x, dim_y, 2]
            nucleus_sizes = np.array(self.rates['nucleus_size'][:2]) / 2
            cytosol_sizes = np.array(self.rates['cytosol_size'][:2]) / 2
        else:  # 3D
            x, y, z = np.meshgrid(np.linspace(0, self.simulation_volume_size[0] - 1, self.simulation_volume_size[0]),
                                np.linspace(0, self.simulation_volume_size[1] - 1, self.simulation_volume_size[1]),
                                np.linspace(0, self.simulation_volume_size[2] - 1, self.simulation_volume_size[2]),
                                indexing='ij')
            positions = np.stack((x, y, z), axis=-1)  # Shape: [dim_x, dim_y, dim_z, 3]
            nucleus_sizes = np.array(self.rates['nucleus_size']) / 2
            cytosol_sizes = np.array(self.rates['cytosol_size']) / 2
        # Calculating normalized squared distance for nucleus and cytosol
        center_of_box = self.center_of_box[:positions.shape[-1]]  # Adjust center according to dimensions
        normalized_sq_dist_nucleus = np.sum(((positions - center_of_box) / nucleus_sizes)**2, axis=-1)
        normalized_sq_dist_cytosol = np.sum(((positions - center_of_box) / cytosol_sizes)**2, axis=-1)
        # Creating masks based on the normalized squared distances
        self.nucleus_mask = normalized_sq_dist_nucleus <= 1
        self.cytosol_mask = normalized_sq_dist_cytosol <= 1
        self.cytosol_mask &= ~self.nucleus_mask  

    def is_within_nucleus(self, entity=None,pos=None,nucleus_diameter=None):
        dimension = 2 if self.simulation_type == '2D' else 3
        if entity is not None:
            pos = np.array(entity['position'][:dimension])
            nucleus_size = np.array(self.rates['nucleus_size'][:dimension]) / 2
        else:
            pos = np.array(pos[:dimension])
            nucleus_size = nucleus_diameter / 2
        #pos = np.array(entity['position'][:dimension])
        center_of_box = self.center_of_box[:dimension]
        normalized_sq_dist = np.sum(((pos - center_of_box) / nucleus_size) ** 2)
        return normalized_sq_dist <= 1
    
    def is_within_cytosol(self, entity):
        dimension = 2 if self.simulation_type == '2D' else 3
        pos = np.array(entity['position'][:dimension])
        cytosol_size = np.array(self.rates['cytosol_size'][:dimension]) / 2
        center_of_box = self.center_of_box[:dimension]
        normalized_sq_dist = np.sum(((pos - center_of_box) / cytosol_size) ** 2)
        is_in_cytosol = normalized_sq_dist <= 1
        return is_in_cytosol and not self.is_within_nucleus(entity)
    
    def move_particle(self, entity, rate):
        dimension = 2 if self.simulation_type == '2D' else 3
        displacement = np.random.normal(scale=np.sqrt(rate), size=dimension)
        current_position = np.array(entity['position'][:dimension])
        new_position = current_position + displacement
        temp_entity = {'position': new_position}
        # RNA specific logic
        if entity['entity_type'] == 'RNA':
            # Check if the RNA is within the cytosol 
            if entity['in_cytosol'] and self.is_within_cytosol(temp_entity):
                entity['position'] = new_position.tolist()
            # Check if the RNA is within the nucleus
            elif not entity['in_cytosol'] and self.is_within_nucleus(temp_entity):
                entity['position'] = new_position.tolist()
            # If RNA is trying to cross from the nucleus to the cytosol without proper transport, ignore
        # Protein specific logic
        elif entity['entity_type'] == 'Protein':
            # Check user permission for protein movement in the nucleus
            if self.movement_protein_into_nucleus:
                # Allow movement in both nucleus and cytosol
                if self.is_within_cytosol(temp_entity) or self.is_within_nucleus(temp_entity):
                    entity['position'] = new_position.tolist()
            else:
                # Restrict movement to the cytosol only
                if self.is_within_cytosol(temp_entity):
                    entity['position'] = new_position.tolist()
                # If the new position is outside the cytosol (and in the nucleus), ignore the move
        return entity
    
    def is_near_nuclear_envelope(self, rna_position):
        dimension = 2 if self.simulation_type == '2D' else 3
        # Adjust the nucleus and shrunk nucleus sizes for the dimension
        nucleus_size = np.array(self.rates['nucleus_size'][:dimension])
        shrunk_nucleus_size = nucleus_size - 2 * self.transport_zone_threshold  # Shrink in all directions
        # Ensure that the shrunk size does not go to zero or negative
        shrunk_nucleus_size = np.clip(shrunk_nucleus_size, a_min=self.transport_zone_threshold, a_max=None)
        pos = np.array(rna_position[:dimension])
        center_of_box = self.center_of_box[:dimension]
        # Calculate if the RNA is within the actual nucleus ellipsoid
        within_nucleus = np.sum(((pos - center_of_box) ** 2) / ((nucleus_size / 2) ** 2)) <= 1
        # Calculate if the RNA is outside the shrunk nucleus (inner boundary)
        outside_shrunk_nucleus = np.sum(((pos - center_of_box) ** 2) / ((shrunk_nucleus_size / 2) ** 2)) > 1
        return within_nucleus and outside_shrunk_nucleus

    def calculate_rates_and_reactions(self):
        rates = []
        reactions = []
        if not self.TS_state:
            rates.append(self.rates['k_on'])
            reactions.append(('TS_on', None))
        else:
            rates.append(self.rates['k_off'])
            reactions.append(('TS_off', None))
            rates.append(self.rates['k_r'])
            reactions.append(('produce_RNA', None))
        for rna_id, rna_info in self.RNAs.items():
            rates.append(self.rates['gamma_r'])
            reactions.append(('degrade_RNA', rna_id))
            rates.append(self.rates['k_diff_r'])
            reactions.append(('move_RNA', rna_id))
            if rna_info['in_cytosol']:
                rates.append(self.rates['k_p'])
                reactions.append(('produce_Protein', rna_id))
            if not rna_info['in_cytosol'] and self.is_near_nuclear_envelope(np.array(rna_info['position'])):
                rates.append(self.rates['transport_rate'])
                reactions.append(('transport_RNA_to_cytosol', rna_id))
        for protein_id in self.Proteins.keys():
            rates.append(self.rates['gamma_p'])
            reactions.append(('degrade_Protein', protein_id))
            rates.append(self.rates['k_diff_p'])
            reactions.append(('move_Protein', protein_id))
        return rates, reactions

    def execute_reaction(self, reaction, current_time):
        reaction_type, entity_id = reaction
        if reaction_type == 'TS_off':
            self.TS_state = False
        elif reaction_type == 'TS_on':
            self.TS_state = True
        
        elif reaction_type == 'produce_RNA':
            # Assuming transcription site is the initial position for RNA
            new_rna = {
                'id': self.next_rna_id,
                'position': self.transcription_site.tolist(),
                'in_cytosol': False,
                'entity_type': 'RNA',
                'time': current_time,
            }
            self.RNAs[self.next_rna_id] = new_rna
            self.next_rna_id += 1
        elif reaction_type == 'degrade_RNA' and entity_id in self.RNAs:
            del self.RNAs[entity_id]
        elif reaction_type == 'move_RNA':
            if entity_id in self.RNAs:
                rna_info = self.RNAs[entity_id]
                self.move_particle(rna_info, self.rates['k_diff_r'])
        elif reaction_type == 'transport_RNA_to_cytosol':
            if entity_id in self.RNAs:
                rna_info = self.RNAs[entity_id]
                rna_info['in_cytosol'] = True
                # Determine dimensionality based on simulation type
                dimension = 2 if self.simulation_type == '2D' else 3
                # Adjust nucleus radius and center_of_box for the dimension
                nucleus_radius = np.array(self.rates['nucleus_size'][:dimension]) / 2
                center_of_box_dim = self.center_of_box[:dimension]
                # Adjust direction_vector calculation for dimensionality
                direction_vector = np.array(rna_info['position'][:dimension]) - center_of_box_dim
                direction_vector /= np.linalg.norm(direction_vector)  # Normalize the vector
                # Calculate new position considering the dimensionality
                new_position = center_of_box_dim + direction_vector * (nucleus_radius + self.small_distance_outside_nucleus)
                # Ensure the new position has the correct dimensionality
                rna_info['position'][:dimension] = new_position.tolist()
        elif reaction_type == 'produce_Protein' and entity_id in self.RNAs:
            # Proteins are produced at the RNA's current position
            protein_info = self.RNAs[entity_id].copy()
            if protein_info['in_cytosol']:  # Ensure Protein is produced only if RNA is in cytosol
                new_protein = {
                    'id': self.next_protein_id,
                    'position': protein_info['position'],
                    'in_cytosol': True, # Proteins once produced are always in cytosol
                    'entity_type': 'Protein',
                    'time': current_time,
                }
                self.Proteins[self.next_protein_id] = new_protein
                self.next_protein_id += 1    
        elif reaction_type == 'degrade_Protein' and entity_id in self.Proteins:
            del self.Proteins[entity_id]
        elif reaction_type == 'move_Protein' and entity_id in self.Proteins:
            protein_info = self.Proteins[entity_id]
            proposed_new_position = self.move_particle(protein_info, self.rates['k_diff_p'])
            # Since proteins are always in the cytosol, we only update if within cytosol
            if self.is_within_cytosol(proposed_new_position):
                self.Proteins[entity_id] = proposed_new_position
            else:
                # If the new position is outside the cytosol, ignore the move
                pass
    
    def simulate(self):
        current_time = 0
        while current_time < self.total_time:
            # Calculate rates and reactions for the current state
            rates, reactions = self.calculate_rates_and_reactions()
            if not rates:
                # If there are no reactions left, advance to the next significant time point (next save time or total time)
                next_time_point = min(self.next_save_time, self.total_time)
                if current_time < next_time_point:
                    current_time = next_time_point
                else:
                    break  # End simulation if beyond total time and no actions are pending
            else:
                # Determine the time until the next reaction occurs
                time_step = np.random.exponential(1 / sum(rates))
                current_time += time_step
            # Update parameters if the drug application time has been reached and it's not already updated
            if (self.apply_drug==True) and (current_time >= self.drug_application_time) and (self.parameters_updated==False):
                self.update_rates_with_drug_effect()
                self.parameters_updated = True
            # Check if the current time is beyond the burnin time and it's time to save the state
            if current_time >= self.burnin_time and current_time >= self.next_save_time:
                self.save_state(current_time)
                self.next_save_time += self.frame_rate  # Schedule the next state save
            # Execute any reactions that were supposed to happen at this time
            if rates:
                reaction_index = np.random.choice(len(rates), p=np.array(rates) / sum(rates))
                reaction = reactions[reaction_index]
                self.execute_reaction(reaction, current_time)

    def run(self):
        self.simulate()
        self.generate_masks()
        #RNAs_positions = self.RNA_trajectories
        #Proteins_positions = self.Protein_trajectories 
        return {
            #'RNAs': RNAs_positions,
            #'Proteins': Proteins_positions,
            'RNA_trajectories': self.RNA_trajectories,
            'Protein_trajectories': self.Protein_trajectories,
            'TS_trajectory': self.TS_trajectory,
            'nucleus_mask': self.nucleus_mask,
            'cytosol_mask': self.cytosol_mask,
            'time_steps': self.time_steps,  # Include time_steps in the returned dictionary
        }
    
def simulate_gene_expression(params_seed):
    params, seed = params_seed
    np.random.seed(seed)  # Aunique seed for each process
    simulator = GeneExpressionSimulator(params)
    return simulator.run()

def get_counts(trajectories):
    time_steps = np.array(trajectories['time_steps'])
    # Initialize counts for RNA and Protein, both in cytosol and nucleus, as NumPy arrays
    RNA_nucleus = np.zeros(len(time_steps))
    RNA_cytosol = np.zeros(len(time_steps))
    Protein_nucleus = np.zeros(len(time_steps))
    Protein_cytosol = np.zeros(len(time_steps))
    # Function to find the index of the closest time step
    def find_closest_time_step_index(target_time):
        return np.argmin(np.abs(time_steps - target_time))
    # Process RNA and Protein trajectories
    for entity_type in ['RNA', 'Protein']:
        entity_trajectories = trajectories[f'{entity_type}_trajectories']
        for entity_list in entity_trajectories.values():
            for snapshot in entity_list:
                time_step_index = find_closest_time_step_index(snapshot['time'])
                if snapshot['in_cytosol']:
                    location_array = 'cytosol'
                else:
                    location_array = 'nucleus'
                if entity_type == 'RNA':
                    if location_array == 'cytosol':
                        RNA_cytosol[time_step_index] += 1
                    else:
                        RNA_nucleus[time_step_index] += 1
                else: # Protein
                    if location_array == 'cytosol':
                        Protein_cytosol[time_step_index] += 1
                    else:
                        Protein_nucleus[time_step_index] += 1
    
    # Calculate total Protein counts by summing cytosol and nucleus counts
    total_Protein = Protein_cytosol + Protein_nucleus
    return time_steps, RNA_nucleus, RNA_cytosol, total_Protein




# def plot_distribution(data_arrays, time_points, bin_min, bin_max, n_bins, list_colors, super_titles=None, drug_application_time=None,folder_outputs=None):
#     """
#     Plots the distribution of protein levels at specified time points for multiple datasets and
#     displays a larger summary plot showing mean and standard deviation below all histograms.

#     Parameters:
#     - data_arrays: list of numpy.ndarray, each 2D array where columns represent different time points.
#     - time_points: list or array of integers, each corresponding to a column in each data array.
#     - bin_min: float, minimum value for the histogram bins.
#     - bin_max: float, maximum value for the histogram bins.
#     - n_bins: int, number of bins in the histogram.
#     - list_colors: list of str, colors for the histogram bars for each dataset.
#     - super_titles: list of str, titles for each row of plots.
#     """
#     n_rows = len(data_arrays)
#     n_cols = len(time_points)
#     species =['R_n', 'R_c','P' ]
#     # Setup the figure and axes grid
#     fig = plt.figure(figsize=(30, 4 * n_rows))  # Adjusted height for additional plot at the bottom
#     grid = plt.GridSpec(n_rows + 1, n_cols, figure=fig)  # Add an extra row for the summary plot
#     def bins_histogram(data):
#         if data.max() > 60:
#             step_bins =2
#         else:
#             step_bins=1
#         bins = np.arange(np.floor(data.min()), np.ceil(data.max()), step_bins)
#         return bins
#     # Plot histograms
#     axes = []  # Store axes for modifying layout later if needed
#     for row, data in enumerate(data_arrays):
#         row_axes = []
#         for col, time in enumerate(time_points):
#             ax = fig.add_subplot(grid[row, col])
#             # Select the column of data corresponding to the current time point
#             data_at_time = data[:, time]
#             bins = bins_histogram(data_at_time)
#             # Create histogram
#             ax.hist(data_at_time, bins=bins, color=list_colors[row]) # np.linspace(bin_min, bin_max, n_bins)
#             # Set title for each subplot
#             if row==0:
#                 ax.set_title(f'{time} sec', fontsize=16)
#             if col == 0:
#                 ax.set_ylabel('Frequency ('+f"${species[row]}$" +')', fontsize=16 )
#             row_axes.append(ax)
#             if row == len(data_arrays) - 1:
#                 ax.set_xlabel('Counts', fontsize=16)
#         axes.append(row_axes)

#     plot_name = 'snapshots_simulated_data.jpg'
#     if folder_outputs is not None:
#         plt.savefig(folder_outputs.joinpath(plot_name), dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.5)
#     else:
#         plt.savefig(folder_outputs.joinpath(plot_name), dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.5)  
#     plt.show()

def plot_distribution(data_arrays, time_points, bin_min, bin_max, n_bins, list_colors, super_titles=None, drug_application_time=None, folder_outputs=None):
    """
    Plots the distribution of protein levels at specified time points for multiple datasets and
    ensures uniform bin sizing and ranges across all histograms.

    Parameters:
    - data_arrays: list of numpy.ndarray, each 2D array where columns represent different time points.
    - time_points: list or array of integers, each corresponding to a column in each data array.
    - bin_min: float, minimum value for the histogram bins.
    - bin_max: float, maximum value for the histogram bins.
    - n_bins: int, number of bins in the histogram.
    - list_colors: list of str, colors for the histogram bars for each dataset.
    - super_titles: list of str, titles for each row of plots.
    - drug_application_time: float, optional, time at which drug is applied (not shown in this plot).
    - folder_outputs: pathlib.Path or str, directory to save the figure.
    """
    n_rows = len(data_arrays)
    n_cols = len(time_points)
    species = ['R_n', 'R_c', 'P']
    font_size = 22
    # Calculate bins outside of the loop to ensure consistency
    bins = np.linspace(bin_min, bin_max, n_bins+1)
    # Setup the figure and axes grid
    fig = plt.figure(figsize=(30, 4 * n_rows))
    grid = plt.GridSpec(n_rows, n_cols, figure=fig)
    # Plot histograms
    for row, data in enumerate(data_arrays):
        for col, time in enumerate(time_points):
            ax = fig.add_subplot(grid[row, col])
            data_at_time = data[:, col]
            ax.hist(data_at_time, bins=bins, color=list_colors[row], edgecolor='black')
            # Set title and labels
            if row == 0:
                ax.set_title(f'{time} sec', fontsize=font_size)
            if col == 0:
                ax.set_ylabel(f'Frequency (${species[row]}$)', fontsize=font_size)
            if row == n_rows - 1:
                ax.set_xlabel('Counts', fontsize=font_size)
    # Save the figure if folder is specified
    if folder_outputs is not None:
        plot_name = 'snapshots_simulated_data.jpg'
        plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.5)
    plt.show()


def simulate_cell_2d (trajectories, list_times, simulation_volume_size, mask_cytosol, mask_nucleus,protein_scaling_factor=2.5 ):
    def get_positions_at_time_step(trajectories, time_step_index=None):
        time_steps = trajectories['time_steps']
        RNA_trajectories = trajectories['RNA_trajectories']
        Protein_trajectories = trajectories['Protein_trajectories']
        def extract_positions_at_index(entity_trajectories, target_index):
            positions = []
            for entity_list in entity_trajectories.values():
                for snapshot in entity_list:
                    if snapshot['time'] == time_steps[target_index]:
                        positions.append(snapshot['position'])
            return positions
        if time_step_index is not None and time_step_index < len(time_steps):
            RNA_positions = extract_positions_at_index(RNA_trajectories, time_step_index)
            Protein_positions = extract_positions_at_index(Protein_trajectories, time_step_index)
        else:
            RNA_positions, Protein_positions = [], []
        return RNA_positions, Protein_positions
    


    def generate_background_image(mask_nucleus, mask_cytosol, simulation_volume_size,Protein_positions, protein_scaling_factor=protein_scaling_factor):    
        threshold_background = 40   # Lower noise in the background
        threshold_nucleus = 150      # Higher noise inside the nucleus
        threshold_cytosol = 80         # Higher noise inside the cytosol
        # Generate random noise for the entire image
        random_values = np.random.rand(simulation_volume_size[0], simulation_volume_size[1])
        image_RNA_channel = np.zeros((simulation_volume_size[0], simulation_volume_size[1]))
        #image_protein_channel = np.zeros((simulation_volume_size[0], simulation_volume_size[1]))
        # Define masks
        background_mask = (mask_nucleus == 0) & (mask_cytosol == 0)
        nucleus_mask = mask_nucleus == 1
        cytosol_mask = mask_cytosol == 1
        # Apply different thresholds to different regions
        image_RNA_channel[background_mask] = random_values[background_mask] * threshold_background
        image_RNA_channel[nucleus_mask] = random_values[nucleus_mask] * threshold_nucleus
        image_RNA_channel[cytosol_mask] = random_values[cytosol_mask] * threshold_cytosol
        image_RNA_channel = gaussian_filter(image_RNA_channel, sigma=4)

        # generate the protien channel by using the protein count 
        #number_proeins= len(Protein_positions)
        image_protein_channel = np.zeros((simulation_volume_size[0], simulation_volume_size[1]))
        image_protein_channel[background_mask] = random_values[background_mask] * threshold_background
        image_protein_channel[nucleus_mask] = random_values[nucleus_mask] * threshold_background
        image_protein_channel[cytosol_mask] = random_values[cytosol_mask] * threshold_cytosol #random_values[cytosol_mask] * (threshold_cytosol+(number_proeins*protein_scaling_factor))
        # clipp at 255
        image_protein_channel = gaussian_filter(image_protein_channel, sigma=4)
        #image_protein_channel = np.clip(image_protein_channel, 0, 255)
        # generate the protien channel by 
        return image_RNA_channel, image_protein_channel

    def add_gaussian_spots(image_array, positions, amplitude=200, sigma=1):
        """
        Adds 2D Gaussian spots to the image at specified positions.

        Parameters:
        - image_array: 2D NumPy array representing the image.
        - positions: List of (x, y) positions where spots will be added.
        - amplitude: Peak intensity of the Gaussian spots.
        - sigma: Standard deviation of the Gaussian (controls the spread).
        """
        # Create a coordinate grid for the entire image
        x_grid, y_grid = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))
        # Add Gaussian spots for each position
        for pos in positions:
            x0, y0 = pos[:2]
            # Create a 2D Gaussian
            #amplitude_spot = np.random.normal(amplitude, 0.1*amplitude)
            gaussian = amplitude * np.exp(-(((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2)))
            # Add the Gaussian to the image
            image_array += gaussian
            # clipp the values to avoid overflow at 255
            image_array = np.clip(image_array, 0, 255)
        return image_array.astype(np.uint8)
    
    simulate_cell = np.zeros((len(list_times), simulation_volume_size[0], simulation_volume_size[1], 2))
    for i,time_step_index in enumerate(list_times):
        # Get RNA positions at the specified time step
        RNA_positions, Protein_positions = get_positions_at_time_step(trajectories, time_step_index)
        # Generate the background image
        image_RNA_channel, image_protein_channel = generate_background_image(mask_nucleus, mask_cytosol, simulation_volume_size,Protein_positions,protein_scaling_factor)
        # Add Gaussian spots representing RNA
        image_RNA_channel = add_gaussian_spots(image_RNA_channel, RNA_positions, amplitude=100, sigma=1)
        # merge the two channels to end with a 2D image

        image_protein_channel = add_gaussian_spots(image_protein_channel, Protein_positions, amplitude=100, sigma=1)

        simulate_cell[i] = np.stack((image_RNA_channel, image_protein_channel), axis=-1)
    return simulate_cell


# -----------------------------
# Segmentation using Thresholding
# -----------------------------

def cell_segmentation (image, sigma=10):
    # Apply multi-level thresholding to segment the image into background, cytosol, and nucleus
    simulated_image_gaussian = gaussian_filter(image, sigma=10)
    thresholds = threshold_multiotsu(image, classes=3)
    # Generate regions based on thresholds
    regions = np.digitize(simulated_image_gaussian, bins=thresholds)
    # Extract masks for nucleus and cytosol
    nucleus_mask_segmented = regions == 2
    cytosol_mask_segmented = regions == 1
    cytosol_mask_segmented = binary_dilation(cytosol_mask_segmented, iterations=1)
    return nucleus_mask_segmented, cytosol_mask_segmented

# -----------------------------
# Particle Detection with TrackPy
# -----------------------------

# Detect particles in the simulated image

def detect_particles (image,nucleus_mask_segmented,cytosol_mask_segmented,diameter=5, minmass=20) :
    f = tp.locate(image, diameter=diameter, minmass=minmass, maxsize=diameter+2, separation=2, noise_size=0.5,percentile=60, invert=False)
    # Assign particles to regions based on segmentation
    x_positions = f['x'].values
    y_positions = f['y'].values
    # Round positions to nearest integer indices
    x_indices = np.round(x_positions).astype(int)
    y_indices = np.round(y_positions).astype(int)
    # Ensure indices are within image bounds
    x_indices = np.clip(x_indices, 0, image.shape[1] - 1)
    y_indices = np.clip(y_indices, 0, image.shape[0] - 1)
    # Using your segmentation masks 'nucleus_mask_segmented' and 'cytosol_mask_segmented'
    # Assign region labels to particles
    particle_regions = np.full(len(f), 'background', dtype=object)
    particle_regions[nucleus_mask_segmented[y_indices, x_indices]] = 'nucleus'
    particle_regions[cytosol_mask_segmented[y_indices, x_indices]] = 'cytosol'
    # Add the region information to the DataFrame
    f['region'] = particle_regions
    # Separate particles in nucleus and cytosol
    particles_in_nucleus = f[f['region'] == 'nucleus']
    particles_in_cytosol = f[f['region'] == 'cytosol']
    return particles_in_nucleus, particles_in_cytosol

def protein_intensity_quantification(simulated_image_protein, cytosol_mask_segmented):
    # multiply the image by the mask to get the intensity of the protein in the cytosol. calculate the mean intensity inside the mask not considering zeros.CLD_CONTINUED
    protein_cytosol = simulated_image_protein * cytosol_mask_segmented
    protein_cytosol = protein_cytosol[protein_cytosol > 0]
    mean_intensity_cytosol = np.mean(protein_cytosol)
    # calculate the mean intensity of the protein in the nucleus
    return mean_intensity_cytosol


def plot_simulation_results(simulate_cell, list_times, mask_cytosol, mask_nucleus, particles_in_nucleus_list, particles_in_cytosol_list, plot_detection=False, channels_to_plot='both', save_figure=False, folder_outputs=None, plot_name='simulation_spatial_model.png'):
    """
    Plots simulation results with subplots where each column is a time point,
    and each row corresponds to a selected channel.

    Parameters:
    - simulate_cell: numpy array of shape (T, Y, X, C)
    - list_times: list of time point indices
    - mask_cytosol: numpy array of shape (Y, X) for the cytosol mask
    - mask_nucleus: numpy array of shape (Y, X) for the nucleus mask
    - particles_in_nucleus_list: list of DataFrames or dicts with detected particles in the nucleus for each time point
    - particles_in_cytosol_list: list of DataFrames or dicts with detected particles in the cytosol for each time point
    - plot_detection: boolean flag to plot masks and detected spots (default: False)
    - channels_to_plot: 'both', 'RNA', or 'Protein' to select which channels to display
    - save_figure: boolean, if True, saves the figure to the specified folder
    - folder_outputs: pathlib.Path or str, the output directory for saving the figure
    - plot_name: str, the filename under which the plot will be saved
    """
    font_size = 22
    channel_indices = {'RNA': [0], 'Protein': [1], 'both': [0, 1]}
    channels = channel_indices[channels_to_plot]
    num_rows = len(channels)
    fig, axs = plt.subplots(num_rows, len(list_times), figsize=(len(list_times) * 4, num_rows * 4))
    
    if len(list_times) == 1:
        axs = np.array(axs).reshape(num_rows, -1)

    for idx, time_index in enumerate(list_times):
        for i, channel in enumerate(channels):
            image_channel = simulate_cell[idx, :, :, channel]
            axs_row = axs[i, idx] if num_rows > 1 else axs[idx]
            axs_row.imshow(image_channel, cmap='gray', vmax=255)
            axs_row.axis('off')
            if idx == 0:
                channel_name = 'RNA Channel' if channel == 0 else 'Protein Channel'
                axs_row.set_ylabel(channel_name, fontsize=font_size)
            if i == 0:
                axs_row.set_title(f'Time {time_index}', fontsize=font_size)
            if plot_detection:
                particles_in_nucleus = particles_in_nucleus_list[idx]
                particles_in_cytosol = particles_in_cytosol_list[idx]
                axs_row.scatter(particles_in_nucleus['x'], particles_in_nucleus['y'], c='red', s=50, label='Nucleus Particles', edgecolors='none')
                axs_row.scatter(particles_in_cytosol['x'], particles_in_cytosol['y'], c='yellow', s=50, label='Cytosol Particles', edgecolors='none')
                axs_row.contour(mask_nucleus, levels=[0.5], colors='red', linewidths=3, linestyles='solid')
                axs_row.contour(mask_cytosol, levels=[0.5], colors='yellow', linewidths=3, linestyles='solid')
                if idx == 0 and i == 0:
                    axs_row.legend(loc='upper right', fontsize=12)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if save_figure:
        if folder_outputs is not None:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
        else:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()




def plot_time_courses_simulation(time_steps,mean_RNA_nucleus,mean_RNA_cytosol, mean_total_Protein, std_RNA_nucleus, std_RNA_cytosol, std_total_Protein,species_colors, drug_application_time=None, max_y_val=None,folder_outputs=None,save_figure=False):
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, mean_RNA_nucleus, label='RNA in Nucleus', color=species_colors['R_n'],lw=4)
    plt.plot(time_steps, mean_RNA_cytosol, label='RNA in Cytosol', color=species_colors['R_c'],lw=4)
    plt.plot(time_steps, mean_total_Protein, label='Protein', color=species_colors['P'],lw=4)
    # plotting the std      
    plt.fill_between(time_steps, mean_RNA_cytosol - std_RNA_cytosol, mean_RNA_cytosol + std_RNA_cytosol, color=species_colors['R_c'], alpha=0.1)
    plt.fill_between(time_steps, mean_RNA_nucleus - std_RNA_nucleus, mean_RNA_nucleus + std_RNA_nucleus, color=species_colors['R_n'], alpha=0.1)    
    plt.fill_between(time_steps, mean_total_Protein - std_total_Protein, mean_total_Protein + std_total_Protein, color=species_colors['P'], alpha=0.1)
    if not drug_application_time is None:
        plt.axvline(x=drug_application_time, color='k', linestyle='--', label= r'$t_{drug}$',lw=1.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Counts')
    plt.title('Mean Counts of RNA and Protein over Time')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    if max_y_val is not None:
        plt.ylim([0, max_y_val])
    plot_name = 'mean_counts_simulated_cell.png'
    if save_figure:
        if folder_outputs is not None:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
        else:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)  
    plt.show()