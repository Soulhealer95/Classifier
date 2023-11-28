# Classifier class that uses sklearn to work on a well defined dataset and compare two classifier algorithms
import openpyxl as op
import numpy as np


class Classifer:

	def __init__(self, test_data_csv, train_data_csv):
		self.test_data_raw = op.load_workbook(test_data_csv)
		self.train_data_raw = op.load_workbook(train_data_csv)
		self.test_data = []
		self.train_data = []
		

	def convert_data(self):
		#stub
	def train_data(self):
		#stub
	def predict(self):
		#stub
	def assess(self):
		#stub
	def get_error(self):
		#stub 

