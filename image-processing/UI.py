import math
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import warnings
from linefinder import Linefinder
warnings.filterwarnings("ignore")

class UserInterface():

    def __init__(self):
        # D:\Tom\Documents\Physics\PHYS355\phys355_code\scans_75dpi <- my own personal path
        print("*** Welcome to the User Interface for evaluating Machine Direction line severity in TFP nonwoven samples! *** \nPlease ensure sample scans are pre-arranged in folders, saved as .jpeg and are of the same material type and areal weight class. You will be led through a series of prompts for which responses must be entered, before results are delivered. Type CTRL+C into the terminal to quit the program at any point. \n") #Type 'back' if you wish to return to a previous question and change your answer.
        self.information()
        self.path = Path(self.get_path()) 
        self.material_types = {1: 'Carbon Veil', 2: 'Metal-Coated Carbon Veil'}  
        self.files = []
        

    def information(self):
        print('***Please respond yes or no to the following question***')
        question = input('Would you like an additional explanation of how the code works, or what the results of this program will tell you? \nNote: There are many prompts, throughout the process that allow you to gain more information and insight to the code that are more relevant to specific processes. \nResponse: ').lower()
        if not question in ('yes', 'no'):
            self.yesno_error()
            self.information()
        if question == 'yes':
            works_or_results = input('\nWould you like information on the working of the code, or on the results? \nPlease respond "workings", "results", or "both". \nResponse: ').lower()
            if not works_or_results in ('workings', 'results', 'both'):
                print('Response not recognised, please try again \n')
                self.information()
            if works_or_results == 'results':
                print('The results of the code will be a number score, a score out of 10, and a pass or failed mark. All of these results are subject to change based on the parameters chosen.')
                print('The number score will show the least variation. It will change if the same sample is blurred using a different sigma value for the guassian blur, but only that.')
                print('The number score is the exact value of the mean peak prominence across the entire sample. This mean prominence represents the difference between light intensity maxima - the machine direction lines and the baseline value of the sample.')
                print('The score out of ten is subject to whether or the sample is being compared to all samples of the same type, or just samples in a similar areal weight class.')
                print('This gives a score, approximately between zero and ten for the severity of machine direction lines in the current sample, compared to similar samples.')
                print('note: the best and worst samples that were sent to Lancaster University defined a score of ten, and zero respectively. Therefore testing samples that give results not between zero and ten are samples that lie outside of this range.')
                print('Testing against all samples of the same material - or sample type - is recommended, as this gives a better appreciation for the visual severity of MD lines, which does not depend on areal weight.')
                print('The pass fail mark depends on whether the number score is above or below the defined value, that yiu will be later prompted to enter.')
                print('Our recommendations are included in this prompt, and you can also respond "info" for more information at that time.')
                print('Plots of detected lines can also be viewed, however this was implemented more for the testing process. If a visual representation of machine direction lines is required, just looking at the sample is better!')
                print('For more information, please contact the code authors through the github repository: https://github.com/msychung/Industrial-Group-Project \n \n')

            if works_or_results == 'workings':
                print('The linefinding process relies primarily on two modules: sci-kit image, and scipy signal.')
                print('The linefinding process is also dependent on several parameters, as will be prompted to you as you go through the code.')
                print('The key dependent of the linefinding process, is sample type - the material from which the sample is made. Only one type can be entered at a time.')
                print('This is due to the fact that the different materials lead to very different results, and so they cannot be directly compared.')
                print('Sci-kit image is used to read the sample in as a numpy array. It then undergoes a gaussian blur to minimise noise.')
                print('Various programs within the scipy signal module are then used to find all of the peaks in all of the individual rows of the sample.')
                print('Once all of the peaks in the sample have been found, the prominence of the peaks are calculated, to judge the difference between the peak and the baseline of the sample.')
                print('The bigger this difference, the more visbile the corresponding machine direction line is. The mean of all of the peak prominences is then calculated.')
                print('This mean value is used to determine the severity of machine direction lines in a sample.')
                print('For more information, please contact the code authors, or view the code, through the github repository: https://github.com/msychung/Industrial-Group-Project \n \n')

            if works_or_results == 'both':
                print('The results of the code will be a number score, a score out of 10, and a pass or failed mark. All of these results are subject to change based on the parameters chosen')
                print('The number score will show the least variation. It will change if the same sample is blurred using a different sigma value for the guassian blur, but only that')
                print('The number score is the exact value of the mean peak prominence across the entire sample. This mean prominence represents the difference between light intensity maxima - the machine direction lines and the baseline value of the sample')
                print('The score out of ten is subject to whether or the sample is being compared to all samples of the same type, or just samples in a similar areal weight class')
                print('This gives a score, approximately between zero and ten for the severity of machine direction lines in the current sample, compared to similar samples')
                print('note: the best and worst samples that were sent to Lancaster University defined a score of ten, and zero respectively. Therefore testing samples that give results not between zero and ten are samples that lie outside of this range')
                print('Testing against all samples of the same material - or sample type - is recommended, as this gives a better appreciation for the visual severity of MD lines, which does not depend on areal weight')
                print('The pass fail mark depends on whether the number score is above or below the defined value, that yiu will be later prompted to enter')
                print('Our recommendations are included in this prompt, and you can also respond "info" for more information at that time')
                print('Plots of detected lines can also be viewed, however this was implemented more for the testing process. If a visual representation of machine direction lines is required, just looking at the sample is better!')
                
                print('\nThe linefinding process relies primarily on two modules: sci-kit image, and scipy signal')
                print('The linefinding process is also dependent on several parameters, as will be prompted to you as you go through the code.')
                print('The key dependent of the linefinding process, is sample type - the material from which the sample is made. Only one type can be entered at a time.')
                print('This is due to the fact that the different materials lead to very different results, and so they cannot be directly compared.')
                print('Sci-kit image is used to read the sample in as a numpy array. It then undergoes a gaussian blur to minimise noise')
                print('Various programs within the scipy signal module are then used to find all of the peaks in all of the individual rows of the sample')
                print('Once all of the peaks in the sample have been found, the prominence of the peaks are calculated, to judge the difference between the peak and the baseline of the sample')
                print('The bigger this difference, the more visbile the corresponding machine direction line is. The mean of all of the peak prominences is then calculated')
                print('This mean value is used to determine the severity of machine direction lines in a sample')
                print('For more information, please contact the code authors, or view the code, through the github repository: https://github.com/msychung/Industrial-Group-Project \n \n')

    def get_path(self):
        response = input("Please specify the path to the folder containing the sample scan files. Enter 'help' for further explanation: ")

        if response.lower() == 'help':
            print("Using file explorer on Windows, open the folder containing the scans. In the navigation bar at the top, right click on the name of the folder, and then click 'copy address as text'. \nThe path should look something like 'C:\Documents\Bob\SampleScans\Carbon\LowAW'\n")
            return self.get_path()

        elif (not Path(response).is_dir()) or (not response):
            print("Path does not exist, please try again.\n")
            return self.get_path()

        else:
            return response

    def yesno_error(self):
        print("Sorry, that response was not recognised, please enter either 'yes' or 'no', and ensure correct spelling.\n")
        
    def input_sample(self):
        sample_type = input("Type of Sample: ")
        
        if sample_type in [str(n) for n in range(len(self.material_types) + 1)]:
            print("Please ensure samples are scanned at 75dpi, or the next closest possible resolution.\n")
            return int(sample_type)
         
        else:
            print(f"Ensure an integer between 1 and {len(self.material_types)} inclusive is entered, please try again.\n")
            self.input_sample()
    
    def input_grouping(self):
        grouping = input(f"Does the folder contain scans of low or high areal weight sample? Enter 'low' or 'high' if you wish to test samples based upon their areal weight class and type, 'all' if you wish to test samples regardless of areal weight, and just on sample type (recommended) or help for more information: ").lower()
        if grouping in ('high', 'low', 'all'):
            return grouping
        if grouping == 'help':
            print('Samples can either be tested against all samples of the same type, or just samples of a similar areal weight. \nThis will lead to a different result for the out of ten score for the sample')
            print('Testing samples against all of the same type will give more information on the visible appearance of the machine direction lines')
            print('As there is an apparent dependence of machine direction lines on areal weight, comparing samples in a similar areal weight class will give more information on how one sample compares to similar samples.\n')
            self.input_grouping()
        else:
            print("Response not recognised, please respond with either 'low', 'high', 'all', or help.\n")
            self.input_grouping()
    
    def input_view_plot(self):
        view_plot = input("Would you like to view the output plot from the analysis software? ").lower()

        if view_plot in ('yes', 'no'):
            return view_plot
        

        else:
            self.yesno_error()
            self.input_view_plot()

    def input_baseline(self):
        baseline = input('\nWhat severity is required to fail a sample? \nRecommendations: 4 for Carbon Veil, 7.5 for Metal-Coated Carbon Veil, when testing against all samples of the same type \nRespond "info" for more information \nEnter Value: ')    #got rid of divide by 2 for now
        
        if baseline.isnumeric():
            return int(baseline)
        
        elif baseline.lower() == 'info':
            print('The method of finding severity of machine direction lines depends on the difference between the background light intensity value of the sample, and the value of  light intensity peaks - caused by machine direction lines.')
            print('The average of this difference is then taken, across the entire sample. If this average is greater than a given number, the sample fails. If it is lower than the given number, the sample passes')
            print('The above recommendations have been made based on the original samples that were sent to Lancaster University, and so different values may be more informative.')
            print('In order to combat this, if the average value of a sample is close to this " pass/fail baseline", then a warning is shown.')
            print('Changing this value will not change the overall score out of ten, and will just change the pass fail result.')
            self.input_baseline()
        else:
            print("Invalid input. Please ensure a positive number is entered.\n")
            self.input_baseline()

    def input_set_parameters(self):
        set_parameters = input("Would you like to use preset values or set your own parameters? ('yes' for presets, 'no' for custom parameters) ").lower()

        if set_parameters in ('yes', 'no'):
            return set_parameters

        else:
            self.yesno_error()
            self.input_set_parameters()

    def input_sigma(self):
        sigma = input("Set sigma value of Gaussian blur to determine severity of image blurring. Recommended value is 1: ")     #and can be... what? what type of number can they input??

        if sigma.isnumeric():   #currently checks for int, might need to change this to float??
            return int(sigma)
        

        
        else:
            print("Invalid input. Please ensure a number is entered.\n")  
            self.input_sigma()
    
    def input_row(self, scanned):
        row = input("Set row number of pixels to be analysed: ")

        if row.isnumeric():

            if int(row) > len(scanned): 
                print(f'Row number index out of range. Please enter an integer within the bounds 0 to {len(scanned)}.\n')
                self.input_row(scanned)
            
            return int(row)
        
        
        else:
            print("Invalid input. Please ensure a positive whole number is entered.\n")
            self.input_row(scanned)

    def quit_program(self):
        end_program = input("Respond 'quit' if you have finished and would like to quit the program. \nOr, respond 'again' if you would like to run the program again: ").lower()

        if end_program == 'quit':
            sys.exit()

        elif end_program == 'again':
            '''A new instance is created every time this function runs, overwriting the last instance. So no need to empty files[] list.'''
            new_run = UserInterface()
            new_run.main()

        else:
            print('Sorry, that response was not recognised, please try again.\n')
            self.quit_program()
            

    def main(self):
        no_of_files = len(glob.glob(f"{self.path}\*.jpeg"))
        print(f"{no_of_files} files added successfully.")
        
        print("\nWhat type of sample would you like to test? Enter", end=' ')

        for key, value in self.material_types.items():

            if key == len(self.material_types):
                print(f"or {key} for {value}.") 

            else:
                print(f"{key} for {value},", end=' ')

        print("NOTE: Only one type of sample can be tested at a time. In order to test multiple sample types, please re-run the program for each type.\n")

        sample_type = self.input_sample()
        grouping = self.input_grouping()
        baseline = self.input_baseline()
        print("\n**Please respond 'yes' or 'no' to the following questions:**")
        view_plot = self.input_view_plot()
        set_parameters = self.input_set_parameters()

        for i, sample in enumerate(glob.glob(f"{self.path}\*.jpeg")): 
            scanned = io.imread(fname=sample, as_gray=True)
            sample_name = os.path.basename(Path(sample))

            if set_parameters == 'yes':
                sigma = 1
                row = 250
            
            else:
                sigma, row = self.input_sigma(), self.input_row(scanned)

            self.files.append((sample, grouping))
            finder = Linefinder(scanned, sigma, row)

            if view_plot == 'yes':
                print(f'Result for Sample: {sample_name}')
                score = finder.severity(baseline, grouping, sample_type)
                finder.plot_nice(sample_name, score)

            else:
                print(f'Result for Sample: {sample_name}')
                finder.severity(baseline, grouping, sample_type)
            
        self.quit_program()

run = UserInterface()
run.main()