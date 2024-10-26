#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:00:00 2022

@author: luis_aguilera
"""
import sys; from pathlib import Path
src_dir = next(parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir())
sys.path.extend([str(src_dir)])
from imports import * 
current_dir = Path().resolve()

# Directory Setup
folder_outputs = current_dir.joinpath('Figures_Exercise_Constitutive_sem_1M')

folder_outputs.mkdir(parents=True, exist_ok=True)
# Plotting configuration
plt.rcParams.update({ 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 10,})
colors = ['#FBD148', '#6BCB77', '#AA66CC', '#FF6B6B', '#4D96FF']
species_colors = { 'R_n': colors[2], 'R_c': colors[3], 'P': colors[4]}


# Loading observable data
folder_simulated_data = current_dir.joinpath('simulated_data')
simulated_time_points = np.load(folder_simulated_data.joinpath('time_points_snapshots_image.npy'))
simulated_data_protein = np.load(folder_simulated_data.joinpath('snapshots_total_Protein_image.npy'))
simulated_data_Rn = np.load(folder_simulated_data.joinpath('snapshots_RNA_nucleus_image.npy'))
simulated_data_Rc = np.load(folder_simulated_data.joinpath('snapshots_RNA_cytosol_image.npy'))

try:
    chain_length = int(sys.argv[1])
    burnin_time = int(chain_length*0.5)
    run_ms = True
except:
    run_ms = False
min_value_parameters = 0.0001
max_value_parameters = 100                          

# Parameters for the model
#k_on = 1.2
#k_off = 0
k_r = 3
k_p = 0.9
gamma_r = 0.05  # assuming gamma_rn = gamma_rc
gamma_p = 0.65
transport_rate = 1
diffusion_rate = 10  # assuming diffusion_rate_r = diffusion_rate_p
total_simulation_time = 200
nucleus_diameter = 60
number_of_trajectories = 50
cytosol_diameter = 110
model_type = '2D'
inhibition_constant = 0.1
drug_application_time = 120
kt = calculate_effective_kt(D=nucleus_diameter, k_diff_r=diffusion_rate, transport_rate=transport_rate, model_type=model_type)
print('Effective transport rate: ', kt)
max_y_val = 65
parameter_values = {'k_r': k_r, 'k_t': kt, 'k_p': k_p, 'gamma_r': gamma_r, 'gamma_p': gamma_p}
initial_conditions = { 'R_n': 0, 'R_c': 0, 'P': 0}
# Updating parameter values
#parameter_symbols = ['k_{r}', 'k_{t}', 'k_{p}', r'\gamma_{r}', r'\gamma_{p}', '\u03B1']

parameter_symbols = [
    r'$k_{r}$',    # rate constant for reaction r
    r'$k_{t}$',    # rate constant for reaction t
    r'$k_{p}$',    # rate constant for reaction p
    r'$\gamma_{r}$',  # decay constant for r
    r'$\gamma_{p}$',  # decay constant for p
    r'$\alpha$'    # some constant alpha
]

true_parameter_values = [k_r, kt, k_p, gamma_r, gamma_p, inhibition_constant]
time_points = simulated_time_points # np.linspace(5, total_simulation_time, 6).astype(int) # np.array([5,10,20,40,60,80,100,130,150,200]).astype(int)
number_parameters = len(true_parameter_values)




def bg_logprior(theta_untransformed):
    log_prior = -0.5 * np.sum((theta_untransformed - BG_MU) ** 2 / BG_SIGMA ** 2)
    return log_prior

def Loglikelihood(parameters, observations_data_mean, observations_data_sem, drug_application_time=0, total_simulation_time=201, time_points=None):
    parameter_values = {
        'k_r': parameters[0], 'k_t': parameters[1], 'k_p': parameters[2],
        'gamma_r': parameters[3], 'gamma_p': parameters[4], 'inhibition_constant': parameters[5]
    }
    initial_conditions = { 'R_n': 0, 'R_c': 0, 'P': 0}
    inhibited_parameters = {'k_t': parameter_values['k_t'] * parameter_values['inhibition_constant']}
    try:
        time, concentrations_species =simulate_model(parameter_values, 
                                             initial_conditions, 
                                             total_simulation_time, 
                                             simulation_type='continuous', 
                                             burn_in_time=0, 
                                             drug_application_time=drug_application_time, 
                                             inhibited_parameters=inhibited_parameters, )
        if time_points is not None:
            y_R_n = concentrations_species['R_n'][time_points]
            y_R_c = concentrations_species['R_c'][time_points]
            #y_P_max = np.max(concentrations_species['P'])
            #y_P_min = np.min(concentrations_species['P'] )
            y_P = concentrations_species['P'][time_points]
            #y_P = (y_P - y_P_min) / (y_P_max - y_P_min)
            #print('y_P_max:', y_P_max, 'y_P_min:', y_P_min)
            loglikelihood = 0.0
            #weights_protein = 10
            for i in range(len(observations_data_mean[0])):  # Assuming observations[0] is correctly indexed
                loglikelihood -= (np.sum(observations_data_mean[0][i]) - y_P[i] )**2 / (2 * observations_data_sem[0][i]**2)
                loglikelihood -= (np.sum(observations_data_mean[1][i]) - y_R_n[i])**2 / (2 * observations_data_sem[1][i]**2)
                loglikelihood -= (np.sum(observations_data_mean[2][i]) - y_R_c[i])**2 / (2 * observations_data_sem[2][i]**2)
    except:
        loglikelihood = -1e20
        time = None
        concentrations_species = None
    return loglikelihood, time, concentrations_species


def adaptive_metropolis(log_target_pdf, start, chain_len, initial_scale=1.0, rng=np.random.default_rng()):
    pdim = len(start)
    samples = np.zeros((chain_len, pdim))
    log_target_pdfs = np.zeros(chain_len)
    samples[0, :] = start
    log_target_pdfs[0] = log_target_pdf(samples[0, :])
    nacc = 0
    # Initial larger covariance matrix for broader exploration
    cov_matrix = initial_scale * np.eye(pdim)
    print_freq=10000
    pbar = tqdm(total=chain_len)
    for i in range(1, chain_len):
        xpropose = rng.multivariate_normal(samples[i-1, :], cov_matrix)
        logpipropose = log_target_pdf(xpropose)
        logu = np.log(rng.uniform())
        # Accept or reject the proposal
        if logu < (logpipropose - log_target_pdfs[i-1]):
            samples[i, :] = xpropose
            log_target_pdfs[i] = logpipropose
            nacc += 1
        else:
            samples[i, :] = samples[i-1, :]
            log_target_pdfs[i] = log_target_pdfs[i-1]
        # Update progress bar with acceptance rate information
        if i % print_freq == 0:
            #pbar.set_description(f"Step {i}, ar: {nacc / i:.4f}")
            pbar.set_description(f"Step {i}, ar: {nacc / i:.3f}, ll: {log_target_pdfs[i]:.1f}")
            pbar.update(print_freq)
    pbar.close()
    return samples, log_target_pdfs





# normalize the protein concentration  min max and then 0 to 1
#simulated_data_protein = (simulated_data_protein - simulated_data_protein.min()) / (simulated_data_protein.max() - simulated_data_protein.min())

number_cells = simulated_data_protein.shape[0]

# Calculating mean values and saving in list with shape (n_species, n_time_points)
observations_data_dist = [simulated_data_protein , simulated_data_Rn, simulated_data_Rc]
observations_data_mean = [np.mean(simulated_data_protein, axis=0), np.mean(simulated_data_Rn, axis=0), np.mean(simulated_data_Rc, axis=0)]
observations_data_sem = [np.std(simulated_data_protein, axis=0)/np.sqrt(number_cells), np.std(simulated_data_Rn, axis=0)/np.sqrt(number_cells), np.std(simulated_data_Rc, axis=0)/np.sqrt(number_cells)]
observations_data_sd = [np.std(simulated_data_protein, axis=0), np.std(simulated_data_Rn, axis=0), np.std(simulated_data_Rc, axis=0)]

# Initial values for the chain
# parameter_symbols = [ 'k_r', 'k_t', 'k_p', 'gamma_r', 'gamma_p', 'inhibition_constant']
BG_MU = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
#BG_SIGMA = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
BG_SIGMA = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


def bg_prior_sample(n_sample: int, rng=np.random.default_rng()) -> np.ndarray:
    while True:
        prior_sample = rng.normal(loc=BG_MU, scale=BG_SIGMA, size=(n_sample, BG_MU.shape[0]))
        transformed_sample = 10.0**prior_sample[0,:]
        if np.all(transformed_sample >= min_value_parameters) and np.all(transformed_sample <= max_value_parameters) and (transformed_sample[-1]<1):
            return prior_sample
chain_start =  bg_prior_sample(1)[0]
print('Initial guess:', 10.0**chain_start)

# Assuming you have a function to calculate log likelihood and log prior
def negative_log_posterior(theta_untransformed, observations_data_mean, observations_data_sem, drug_application_time, total_simulation_time, time_points):
    theta = 10.0 ** theta_untransformed  # Apply the transformation within the function
    log_prior = bg_logprior(theta_untransformed)  # Assuming you have a function for the log prior
    log_likelihood = Loglikelihood(theta, observations_data_mean, observations_data_sem, drug_application_time, total_simulation_time, time_points)[0]
    return (log_prior + log_likelihood)  # Negative because minimize seeks to minimize the function
# running the MCMC
if run_ms:
    bg_log_target_partial = partial(negative_log_posterior, observations_data_mean=observations_data_mean,observations_data_sem=observations_data_sem, drug_application_time=drug_application_time, total_simulation_time=total_simulation_time, time_points=time_points)
    #bg_chain, bg_logpos = adaptive_metropolis(bg_log_target_partial, start=chain_start, chain_len=chain_length, cov_matrix=np.eye(len(BG_MU)))
    bg_chain, bg_logpos = adaptive_metropolis(bg_log_target_partial, start=chain_start, chain_len=chain_length, initial_scale=0.00001, rng=np.random.default_rng())
    np.save(folder_outputs.joinpath('bg_chain.npy'), bg_chain)
    np.save(folder_outputs.joinpath('bg_logpos.npy'), bg_logpos)
else:
    bg_chain = np.load(folder_outputs.joinpath('bg_chain.npy'))
    bg_logpos = np.load(folder_outputs.joinpath('bg_logpos.npy'))
    chain_length = bg_chain.shape[0]
    burnin_time = int(chain_length*0.1)
print('mean_ll', np.mean(bg_logpos))
    
    
# plot the ODE model with the best fit parameters
best_fit = np.mean(bg_chain[burnin_time:], axis=0)
best_fit = np.round(10.0**( best_fit),3)
print('best parameters:', best_fit)

ll, time, concentrations_species = Loglikelihood(best_fit, observations_data_mean=observations_data_mean,observations_data_sem=observations_data_sem,drug_application_time=drug_application_time,total_simulation_time=total_simulation_time, time_points=time_points)
print(ll)
# solving the model with the true parameters
print('true parameters:', true_parameter_values)
ll, time, concentrations_species_true = Loglikelihood(true_parameter_values, observations_data_mean=observations_data_mean,observations_data_sem=observations_data_sem,drug_application_time=drug_application_time,total_simulation_time=total_simulation_time, time_points=time_points)
print(ll)



# Plotting the ODE model and the experimental data only for variables P and R_n
plt.figure(figsize=(8, 5))
plt.errorbar(time_points, observations_data_mean[0], yerr=observations_data_sd[0], fmt='o', color=species_colors['P'], label='Spatial Model ' + f"${'(P)'}$", markersize=10, lw=1)
plt.errorbar(time_points, observations_data_mean[1], yerr=observations_data_sd[1], fmt='o', color=species_colors['R_n'], label='Spatial Model ' + f"${'(R_n)'}$", markersize=10, lw=1)
plt.errorbar(time_points, observations_data_mean[2], yerr=observations_data_sd[2], fmt='o', color=species_colors['R_c'], label= 'Spatial Model ' + f"${'(R_c)'}$", markersize=10, lw=1)
plt.plot(time, concentrations_species['P'],   color=species_colors['P'], label='Model Fit ' + f"${'(P)'}$", lw=2, linestyle='--')
plt.plot(time, concentrations_species['R_n'], color=species_colors['R_n'], label='Model Fit ' + f"${'(R_n)'}$", lw=2, linestyle='--')
plt.plot(time, concentrations_species['R_c'], color=species_colors['R_c'], label='Model Fit ' + f"${'(R_c)'}$", lw=2, linestyle='--')
if drug_application_time > 0:
    plt.axvline(x=drug_application_time, color='k', linestyle='--', label='Drug Application', lw=2,)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('ODE Model vs mean(Spatial Model)')
plt.legend(bbox_to_anchor=(1.3, 1),loc='upper right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(folder_outputs.joinpath('model_fit.jpg'), dpi=300, bbox_inches="tight")
plt.show()
plt.close()





fig, axs = plt.subplots(1, number_parameters, figsize=(20, 4))  # Adjusted for a horizontal layout
fig.set_tight_layout(True)
for i in range(number_parameters):
    axs[i].set_title(parameter_symbols[i], fontsize=20)
    axs[i].plot(10.0**(bg_chain[burnin_time:, i]), color='lightslategray', lw=2)
fig.savefig(folder_outputs.joinpath("mcmc_trajectories.jpg"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()



# fig, axs = plt.subplots(1, number_parameters, figsize=(20, 4))  # Adjusted for a horizontal layout
# fig.set_tight_layout(True)
# autocorr = lambda data: np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[(len(data)-1):] / (np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[(len(data)-1)])
# for i in range(number_parameters):
#     axs[i].set_title(parameter_symbols[i], fontsize=20)
#     autocorrelation = autocorr(bg_chain[burnin_time:, i])
#     axs[i].plot(autocorrelation, color='lightslategray', lw=3)
# fig.savefig(folder_outputs.joinpath("mcmc_ac.jpg"), dpi=300, bbox_inches="tight")
# plt.show()



# Loop through the parameter indices and create histograms
fig, axs = plt.subplots(1, number_parameters, figsize=(20, 4))  # Adjusted for a horizontal layout
fig.set_tight_layout(True)
for i in range(number_parameters):
    data = 10.0**(bg_chain[burnin_time:, i])  # Extracting the burn-in adjusted chain for the parameter
    # remove extreme values to plot the histogram
    min_value_parameters = np.percentile(data, 0.1)
    max_value_parameters = np.percentile(data, 99.9)   
    data = data[(data>min_value_parameters) & (data<max_value_parameters)]
     # Calculate the mean of the parameter's chain
    mean = np.mean(data) 
    # Calculate the 90% CIs
    ci_lower, ci_upper = np.percentile(data, [5, 95])  # Calculate the 90% CI
    #xlim_lower, xlim_upper = np.percentile(data, [0.01, 99.])  # Calculate the 0.01 and 99.99 percentiles
    # Histogram plot
    axs[i].hist(data, bins=20, color='lightslategray', alpha=0.7)
    axs[i].axvline(mean, color='orangered', lw=4, label='Estimate')
    axs[i].axvline(ci_lower, color='blue', linestyle='--', lw=2, label='90% CI')
    axs[i].axvline(ci_upper, color='blue', linestyle='--', lw=2)
    # axs[i].axvline(true_parameter_values[i], color='green', lw=4, label='True', linestyle='--')  # Uncomment to mark the true parameter value
    # Set x-axis limits based on the extreme percentiles
    axs[i].set_xlim(min_value_parameters, max_value_parameters)
    axs[i].set_xlabel(parameter_symbols[i], fontsize=20)
    axs[i].set_ylabel('Frequency')
    axs[i].legend()
# Adding a title for all plots
fig.tight_layout()
# Save the figure
fig.savefig(folder_outputs.joinpath("mcmc_1D.jpg"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()




# Set the style
sns.set(style="white")
# Enable LaTeX in matplotlib
chain_trunc = 10.0**(bg_chain[burnin_time:, :]) # Remove burn-in samples
fig, axs = plt.subplots(number_parameters, number_parameters, figsize=(15, 15))
fig.tight_layout(pad=2.0)
for i in range(number_parameters):
    for j in range(i):
        axs[i, j].hexbin(chain_trunc[:, j], chain_trunc[:, i], gridsize=30, cmap='plasma', mincnt=1)
        axs[i, j].tick_params(axis='both', which='major', labelsize=12)
        axs[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if j == 0:
            # Use LaTeX for y-labels
            axs[i, j].set_ylabel(parameter_symbols[i], fontsize=20)
        if i == number_parameters - 1:  # This ensures the x labels appear at the bottom row
            axs[i, j].set_xlabel(parameter_symbols[j], fontsize=20)
    for j in range(i + 1, number_parameters):
        axs[i, j].axis('off')
    # Plot histogram
    n, bins, patches = axs[i, i].hist(chain_trunc[:, i], color= (0.140603, 0.021687, 0.566959, 1.0), bins=20)
    axs[i, i].tick_params(axis='both', which='major', labelsize=12)
    # Use scientific notation for y-axis
    axs[i, i].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axs[i, i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[i, i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# Save the figure
fig.savefig(folder_outputs.joinpath("mcmc_2D.jpg"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()




#@title Plotting SSA model
def plotting_stochastic(time, trajectories_species,species_colors,drug_application_time=None,ylim_val=False,save_figure=True,plot_name='ssa.jpg',time_points=None, observations_data_mean=None, observations_data_sem=None):
    def plot_species_trajectories(time, trajectories_species, species_name, color):
        # Extract the trajectories for the species
        trajectories = trajectories_species[species_name]
        # Calculate the mean and standard deviation across all trajectories
        mean_trajectories = np.mean(trajectories, axis=0)
        std_trajectories = np.std(trajectories, axis=0)
        # Plot mean concentration with standard deviation as shaded area
        #plt.plot(time, mean_trajectories, ':', color=color, label='SSA '+ species_name, lw=2)
        plt.fill_between(time, mean_trajectories - std_trajectories, mean_trajectories + std_trajectories, color=color, alpha=0.4, label='SSA ('+ f"${species_name}$" +')')
    plt.figure(figsize=(8, 5))
    # Plot each species
    for species, color in species_colors.items():
        plot_species_trajectories(time, trajectories_species, species, color)
    if not (time_points is None) and not (observations_data_mean is None) and not (observations_data_sem is None):
        # ploting data
        plt.errorbar(time_points, observations_data_mean[0], yerr=observations_data_sem[0], fmt='o', color=species_colors['P'], label='Spatial Model ' + f"${'(P)'}$", markersize=10, lw=1)
        plt.errorbar(time_points, observations_data_mean[1], yerr=observations_data_sem[1], fmt='o', color=species_colors['R_n'], label='Spatial Model ' + f"${'(R_n)'}$", markersize=10, lw=1)
        plt.errorbar(time_points, observations_data_mean[2], yerr=observations_data_sem[2], fmt='o', color=species_colors['R_c'], label= 'Spatial Model ' + f"${'(R_c)'}$", markersize=10, lw=1)
        # plotting model fit
        plt.plot(time, concentrations_species['P'],   color=species_colors['P'], label='Model Fit ' + f"${'(P)'}$", lw=2, linestyle='--')
        plt.plot(time, concentrations_species['R_n'], color=species_colors['R_n'], label='Model Fit ' + f"${'(R_n)'}$", lw=2, linestyle='--')
        plt.plot(time, concentrations_species['R_c'], color=species_colors['R_c'], label='Model Fit ' + f"${'(R_c)'}$", lw=2, linestyle='--')
    # Mark the drug application time
    if not drug_application_time is None:
        plt.axvline(x=drug_application_time, color='k', linestyle='--', label=r'$t_{drug}$', lw=1.5)
    # Set plot details
    plt.xlabel('Time')
    plt.ylabel('Number of Molecules')
    plt.title('Model vs Data')
    #plt.legend(loc='upper right')
    plt.legend(bbox_to_anchor=(1.3, 1),loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    if ylim_val:
        plt.ylim(0,ylim_val)
    if save_figure == True: 
        plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)  
    plt.show()
    plt.close()




def plot_stochastic_and_data(time, trajectories_species, species_colors, drug_application_time=None, ylim_val=False, save_figure=True, plot_name='ssa.jpg', time_points=None, observations_data_mean=None, observations_data_sem=None, concentrations_species=None, folder_outputs=None):
    """
    Plot stochastic simulation results with two subplots: one for RNA counts, one for protein counts.
    In each plot, include the data, model fit, and spatial data.

    Parameters:
    - time: array-like, time points of the simulation.
    - trajectories_species: dict, keys are species names, values are arrays of trajectories.
    - species_colors: dict, keys are species names, values are color codes.
    - drug_application_time: float, time at which drug is applied.
    - ylim_val: float or False, y-axis limit.
    - save_figure: bool, whether to save the figure.
    - plot_name: str, filename to save the figure.
    - time_points: array-like, time points of the observations.
    - observations_data_mean: list of arrays, mean observations for each species.
    - observations_data_sem: list of arrays, SEM of observations for each species.
    - concentrations_species: dict, model fit data, keys are species names.
    - folder_outputs: pathlib.Path or str, directory to save the figure.
    """
    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    # Define RNA and Protein species
    RNA_species = ['R_n', 'R_c']
    Protein_species = ['P']
    # Function to plot species trajectories
    def plot_species_trajectories(ax, time, trajectories_species, species_list, species_colors):
        for species_name in species_list:
            if species_name in trajectories_species:
                color = species_colors.get(species_name, 'black')
                # Extract the trajectories for the species
                trajectories = trajectories_species[species_name]
                if species_name ==Protein_species:
                    # normalize the protein concentration  min max and then 0 to 1
                    trajectories = (trajectories - trajectories.min()) / (trajectories.max() - trajectories.min())
                # Calculate the mean and standard deviation across all trajectories
                mean_trajectories = np.mean(trajectories, axis=0)
                std_trajectories = np.std(trajectories, axis=0)
                # Plot mean concentration with standard deviation as shaded area
                ax.fill_between(time, mean_trajectories - std_trajectories, mean_trajectories + std_trajectories,
                                color=color, alpha=0.4, label=f'SSA (${species_name}$)')
                ax.plot(time, mean_trajectories, color=color, linestyle='-', lw=2)

    # Plot RNA counts
    ax_rna = axs[0]
    plot_species_trajectories(ax_rna, time, trajectories_species, RNA_species, species_colors)

    # Plot Protein counts
    ax_protein = axs[1]
    plot_species_trajectories(ax_protein, time, trajectories_species, Protein_species, species_colors)

    # If observations data is provided, plot on respective subplots
    if time_points is not None and observations_data_mean is not None and observations_data_sem is not None:
        # Plot RNA data on ax_rna
        for idx, species_name in enumerate(RNA_species):
            if species_name in species_colors:
                color = species_colors[species_name]
                ax_rna.errorbar(time_points, observations_data_mean[idx], yerr=observations_data_sem[idx], fmt='o',
                                color=color, label=f'Spatial Model (${species_name}$)', markersize=8, lw=1)

                # Plot model fit if concentrations_species is provided
                if concentrations_species is not None and species_name in concentrations_species:
                    ax_rna.plot(time, concentrations_species[species_name], color=color, label=f'Model Fit (${species_name}$)',
                                lw=2, linestyle='--')

        # Plot Protein data on ax_protein
        idx_protein = len(RNA_species)  # Assuming protein data is after RNA data in the lists
        species_name = 'P'
        if species_name in species_colors:
            color = species_colors[species_name]
            ax_protein.errorbar(time_points, observations_data_mean[idx_protein], yerr=observations_data_sem[idx_protein], fmt='o',
                                color=color, label=f'Spatial Model (${species_name}$)', markersize=8, lw=1)

            # Plot model fit if concentrations_species is provided
            if concentrations_species is not None and species_name in concentrations_species:
                ax_protein.plot(time, concentrations_species[species_name], color=color, label=f'Model Fit (${species_name}$)',
                                lw=2, linestyle='--')

    # Set plot details
    for ax in axs:
        if drug_application_time is not None:
            ax.axvline(x=drug_application_time, color='k', linestyle='--', label=r'$t_{drug}$', lw=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Molecules')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        if ylim_val:
            ax.set_ylim(0, ylim_val)
    # Set titles
    ax_rna.set_title('RNA Counts')
    ax_protein.set_title('Protein Counts')
    plt.tight_layout()
    if save_figure:
        if folder_outputs is not None:
            plt.savefig(folder_outputs.joinpath(plot_name), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
        else:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
    plt.show()
    plt.close()






# Running SSA
parameter_values = {'k_r': best_fit[0], 'k_t': best_fit[1], 'k_p': best_fit[2],
                'gamma_r': best_fit[3], 'gamma_p': best_fit[4], 'inhibition_constant': best_fit[5]}
initial_conditions = { 'R_n': 0, 'R_c': 0, 'P': 0}
inhibited_parameters = {'k_t': parameter_values['k_t'] * parameter_values['inhibition_constant']}
time_ssa, trajectories_species_ssa =simulate_model(parameter_values, 
                                        initial_conditions, 
                                        total_simulation_time=total_simulation_time, 
                                        simulation_type='discrete', 
                                        burn_in_time=0, 
                                        drug_application_time=drug_application_time, 
                                        inhibited_parameters=inhibited_parameters, 
                                        number_of_trajectories=200,)

# Plotting SSA
plotting_stochastic(time_ssa, 
                    trajectories_species_ssa,
                    species_colors,
                    drug_application_time,
                    ylim_val=70,
                    time_points=time_points, 
                    observations_data_mean=observations_data_mean,
                    observations_data_sem=observations_data_sem )



# Plotting the distributions for the SSA and the observed data
fig, axs = plt.subplots(3, len(time_points), figsize=(20, 6))
fig.tight_layout(pad=2.0)
legend_labels = ['SSA', 'Observed']  # Labels for the legend

# Plotting the distributions for the SSA at a given time point
def bins_histogram(data1, data2):
    data = np.concatenate([data1, data2])
    if data.max() > 20:
        step_bins =3
    else:
        step_bins=1
    bins = np.arange(np.floor(data.min()), np.ceil(data.max()), step_bins)
    return bins
for j, time_point in enumerate (time_points):
    for i, species in enumerate(['P', 'R_n', 'R_c']):
        ssa_data = trajectories_species_ssa[species][:,time_point]
        obs_data = observations_data_dist[i][:,j]
        # fix the bins to the max between simulation and observations
        step_bins = bins_histogram(ssa_data, obs_data)
        obs_plot =axs[i,j].hist(ssa_data, bins=step_bins, color='royalblue', histtype='step', lw=4,density=True) #species_colors[species]
        ssa_plots=axs[i,j].hist(obs_data, bins=step_bins,  color='orangered', lw=1.2, alpha=0.5, density=True)
        #axs[i,j].set_title(f"${species}$ at $t = {time_point}$")  #"${parameter_symbols[j]}$"
        axs[i,j].set_title(f"$t = {time_point}$", fontsize=16)  #"${parameter_symbols[j]}$"
        if j == 0:
            axs[i,j].set_ylabel(f"${species}$", fontsize=20)
# Create a legend outside the plot area
fig.legend([ssa_plots[0], obs_plot[2]],     # Plot elements to be included in the legend
           labels=legend_labels,        # Labels for the legend
           loc='center right',          # Position of the legend
           bbox_to_anchor=(1.10, 0.5),
           fontsize=16)   
plt.savefig(folder_outputs.joinpath("ssa_distributions.jpg"), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
plt.close()





fig, axs = plt.subplots(3, len(time_points), figsize=(20, 6))
fig.tight_layout(pad=2.0)
def bins_histogram(data1, data2):
    data = np.concatenate([data1, data2])
    if data.max() > 20:
        step_bins = 3
    else:
        step_bins = 1
    bins = np.arange(np.floor(data.min()), np.ceil(data.max()), step_bins)
    return bins
legend_labels = ['SSA', 'Observed']  # Labels for the legend
legends_for_obs = []
for j, time_point in enumerate(time_points):
    ssa_plots = []
    obs_plots = []
    for i, species in enumerate(['P', 'R_n', 'R_c']):
        ssa_data = trajectories_species_ssa[species][:, time_point]
        obs_data = observations_data_dist[i][:, j]
        # Using the same bins for both SSA data and observations data
        bins = bins_histogram(ssa_data, obs_data)
        # Plotting SSA data as steps
        ssa_counts, bin_edges = np.histogram(ssa_data, bins=bins, density=True)
        ssa_cdf = np.cumsum(ssa_counts) / np.sum(ssa_counts)
        ssa_cdf = np.concatenate([[0], ssa_cdf])
        # Plotting observations data as histograms
        obs_plot = axs[i, j].hist(obs_data, bins=bins, color='orangered', lw=0.5, density=True, cumulative=True,alpha=0.5)
        ssa_plots.append(axs[i, j].step(bins, ssa_cdf, color='royalblue', lw=5)[0])
        if i == 0:
            axs[i, j].set_title(f"$t = {time_point}$", fontsize=16)
        if j == 0:
            axs[i, j].set_ylabel(f"${species}$", fontsize=20)
        if i == 2:
            axs[i, j].set_xlabel('Molecules', fontsize=16)     
# Create a legend outside the plot area
fig.legend([ssa_plots[0], obs_plot[2]],     # Plot elements to be included in the legend
           labels=legend_labels,        # Labels for the legend
           loc='center right',          # Position of the legend
           bbox_to_anchor=(1.10, 0.5),
           fontsize=16)                 # Font size of the legen
plt.savefig(folder_outputs.joinpath("ssa_cdfs_steps.jpg"), dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)
plt.show()
plt.close()
