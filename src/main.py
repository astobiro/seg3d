from models.unet3dmodel import Unet3Dmodel
from generators.data_loader import VolumeDataGenerator	
import sys, os
from utils.utils import Params
from pprint import pprint
import tensorflow as tf

SRC_ABSOLUTE_PATH = "."
sys.path.append(SRC_ABSOLUTE_PATH)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)

def main():
	args = sys.argv
	print(args[1])
	try:
		# Capture the command line arguments from the interface script.
		args = sys.argv
		print(args)
		# Parse the configuration parameters for the ConvNet Model.
		config = args[1]
		load = args[2]

	except:
		print( 'Missing or invalid arguments !' )
		exit(0)

	# Construct, compile, train and evaluate the ConvNet Model.
	model = Unet3Dmodel(config)
	#print(model.savepath)

	model.define_model()

	if load != 1:
		print("Starting training")
		model.fit_model()
	elif load == 1:
		print("Loading weights")
		model.load_best_results()
	else:
		print("load parameter error, terminating program")
		return

	model.evaluate_model_per_patient()

	# model.iou_calc_save()

	# model.save_image_results()
if __name__ == '__main__':
	main()