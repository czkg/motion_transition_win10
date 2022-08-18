import os
from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
from utils.visualizer import Visualizer
import time



if __name__ == '__main__':
	opt = TrainOptions().parse()       # get training options
	dataset = create_dataset(opt)        # create a dataset given opt.dataset_mode and other options
	dataset_size = len(dataset)        # get dataset size

	model = create_model(opt)
	model.setup(opt)                   # regular setup: load and print networks; create schedulers
	total_iters = 0                    # the total number of training iterations
	visualizer = Visualizer(opt)       # create a visualizer the dispaly/save intermediate results


	# outer loop for different epochs
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()     # timer for entire epoch
		iter_data_time = time.time()       # timer for data loading iteration
		epoch_iter = 0                     # the number of training iterations in current epoch, reset to 0 every epoch


		for i, data in enumerate(dataset):   # inner loop within one epoch
			iter_start_time = time.time()    # timer for computation per iteration
			if total_iters % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			total_iters += opt.batch_size
			epoch_iter += opt.batch_size

			model.set_input(data)            # unpack data from dataset and apply preprocessing
			if not model.is_train():         # if this batch of input data is enough for training
				print('skip this batch')
				continue
			model.optimize_parameters()      # calculate loss function, get gradients, update network weights


			if total_iters % opt.print_freq == 0:
				losses = model.get_current_losses()
				t_comp = (time.time() - iter_start_time) / opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
				# if opt.display_id > 0:
				# 	visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

			# cache the latest model every <save_latest_freq> iterations
			if total_iters % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
				model.save_networks('latest')

			iter_data_time = time.time()

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		# update learning rates at the end of every epoch
		model.update_learning_rate()