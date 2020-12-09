# PHYS301_final_project_lazo
Final project for PHYS 301: Computational Physics at WVU. Implementation of a recurrent neural network to automate analysis of space plasma data.

USAGE OVERVIEW
==============
This is a prototype program designed to utilize deep learning for diagnosis of Langmuir probe data.
This version analyzes low-temperature magnetospheric plasma data, and utilizes data downlinked from NASA's Simulation-to-Flight 1 (STF-1) Cubesat.
TO USE THIS PROGRAM: Please execute the main program script, network.py, from a Python IDE or the command line.
	- The program will first prompt you with a menu to select a data file to use. These are pre-processed STF-1 data files; more data files will follow in future versions.
	- To select a file, please enter the integer that corresponds to the desired file, and press 'Enter'.
	- Once a file is selected, network.py will take over and conduct training. The final output of the program will be a plot of the Mean Squared Error (MSE) loss versus the number of times the data has been trained over, or epochs. The epochs can be changing by editing the EPOCHS variable in network.py.
	- The STF-1 data will be analyzed, and stf1.py will output to the console the "uncharacteristic" electron temperaturese and ion densities calculated from that data. These are data points for which plasma parameters lie outside the limits reported in the literature to define an ionospheric plasma, and will be filtered to one extreme of this definition.
	- As the STF-1 data is imported and the synthetic data is generated, a series of Numpy ndarray shapes and Tensorflow Tensor shapes will be output; this is for sanity checks, to ensure the network is behaving as intended.
	- The program will attempt to load a Keras Sequential model from the 'model' folder in the local directory. If none is found or an error occurs, the network will start from scratch and construct a new network. !THIS WILL OVERWRITE ANY SIMILARLY-NAMED FILES IN THE 'model' FOLDER!
	- Once the model is loaded/constructed and compiled, training will begin. Live updates of training progress can be seen as the epochs progrress; you will watch the network iterate over the synthetic data EVAL_INT times, and calculate MSE training and validation losses.
	- Training will conclude, the model will be saved, and 2-D Numpy histograms of the normalized predicted label values versus normalized actual label values will be created; a plot of MSE loss versus epoch will be saved as 'loss_plot.png' to the local directory.
	- Program execution has ended, or will end when the figure windows are closed if run from the command line.

PROGRAM OVERVIEW
================
constants.py: Record of relevant physical and experimental constants and values for magnetospheric plasmas.
		# Physical constants
		Q = 1.602e-19            # Fundamental charge, in C
		M_P = 1.673e-27          # Proton mass, in kg
		M_E = 9.109e-31          # Electron mass, in kg

		mu_o = 16            # Mass number of oxygen, in amu
		m_i = M_P * mu_o     # Ion mass of singly ionized, monatomic oxygen, in kg

		# Probe parameters
		CUTOFF = 2.420e-06  # Physical cutoff electron current for CubeSat, in A
		A_P = 7.7515e-3     # Probe area, in m^2

		# Label parameters
		T_min = 0.1       # Minimum electron temperature for modeling, in eV
		T_max = 8.5       # Maximum electron temperature for modeling, in eV
		Is_min = -2.0e-7  # Minimum ion saturation current for modeling, in A
		Is_max = 8.0e-7   # Maximum ion saturation current for modeling, in A

network.py: Main script for the program; THIS file should be run from the command line or in an IPython terminal. This file imports real plasma data and generates synthetic data. A neural network is built using Tensorflow 2.3.0, and then trained on the synthetic data.
After training, the network will make predictions on the real and synthetic data, saving plots of MSE loss and R-squared values of model fit parameters.
Uses constants.py, Spacebox.py, stf1.py, and r2plot.py.

SpaceBox.py: Object definition class; defines the object which generates one synthetic current-voltage characteristic when __call()__ is invoked.
Uses constants.py and langmuir.py.

stf1.py: Function definitions for analyzing Simulation-to-Flight 1 (STF-!) CubeSat Langmuir probe data.
Uses constants.py.
	read_data(file): Function to import, clean, and convert STF-1 data to SI units in a format useful for the neural network. NOTE: file MUST correspond to one of the pre-processed data files. As of current release, the available files are zero-indexed in a dictionary:
		0 : Jan. 16, 2019
        1 : Feb. 26, 2019
        2 : Apr. 05, 2019
        3 : Apr. 19, 2019
	Inputting a value which does not match these indices will cause the file to default to index 0, and process the data from January 16, 2019.
	More data will be added in the future.
	RETURNS: CCSDS timestamps, voltage time series as 2-D array (one row is one trace), current time series as 2-D array, and integer number of traces in file
	
	analyze_data(vv, ii, traces): Function to calculate necessary plasma parameters to derive electron temperature in eV and ion saturation current in A.
	RETURNS: electron temperature array (one for each trace), ion saturation array, ion density array, floating potential array, ion current linear fit slope array, ion current linear fit intercept array
	
langmuir.py: Function definitions for synthetic data model.
Uses constants.py
	Ii(v, m, b): Function to generate a simple linear ion current from the provided voltage time series with input slope and intercept.
	RETURNS: calculated linear ion current, as an ndarray
	
	Ie(v, Vf, Ies, Te): Function to generate an exponential electron current as a function of the provided voltage time series. Vf is plasma floating potential, Ies is electron sturation current, and Te is plasma electron temperature.
	RETURNS: calculated exponential electron current offset so that Vf is treated as zero, as an ndarray

r2plot.py: Utility function definitions for constructing R-squared Numpy histograms of normalized predicted versus normalized expected label values.
	RSquared(y, y_p): Function to calculate R-squared values of predictions; y and y_p must be Tensorflow eager tensor objects; y is the array of labels, y_p is the array of predicted values.
	RETURNS: Numpy ndarray of calculated R-squared values, R_squared.numpy()
	
	plotResult(fig, ax, y, y_p, label, title, fontsize=14, res=25): Convenience function to plot 2-D Numpy histograms of normalized predicted values versus normalized label values.
	RETURNS: None; Numpy 2dhistogram() plot generated with matplotlib.pyplot
