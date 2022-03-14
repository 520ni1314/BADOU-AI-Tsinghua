import numpy as np
import myalexnet as mn
from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
import cv2
import utils

def generate_arrays_from_file(lines, batch_size):
	n=len(lines)
	i=0
	while 1:
		x_train=[]
		y_train=[]
		for b in range(batch_size):
			if i==0:
				np.random.shuffle(lines)
			name = lines[i].split(";")[0]
			img = cv2.imread(r"./data/image/train/" + name)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			img = img/255
			x_train.append(img)
			y_train.append(lines[i].split(";")[1])
			i = (i+1)%n
		x_train = utils.resize_image(x_train,(224,224))
		x_train = x_train.reshape(-1,224,224,3)
		y_train = np_utils.to_categorical(np.array(y_train),num_classes=2)
		yield (x_train,y_train)

if __name__=="__main__":
	logdir = "logs/"
	with open("data/dataset.txt") as f:
		lines = f.readlines()
	np.random.seed(10101)
	np.random.shuffle(lines)
	np.random.seed(None)

	num_val = int(len(lines)*0.1)
	num_train = len(lines) - num_val

	model = mn.MyAlexNet()
	checkpoint_period1=ModelCheckpoint(
		logdir + "ep{epoch:03d-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
		monitor='acc',
		save_weights_only=False,
		save_best_only=True,
		period=3
	)

	reduce_lr = ReduceLROnPlateau(
		monitor='acc',
		factor=0.5,
		patience=3,
		verbose=1
	)

	early_stopping = EarlyStopping(
		monitor='val_loss',
		min_delta=0,
		patience=10,
		verbose=1
	)

	model.compile(loss='categorical_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=['accuracy'])
	batch_size = 128

	print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
	model.fit_generator(generate_arrays_from_file(lines[:num_train],batch_size),
						steps_per_epoch=max(1, num_train//batch_size),
						validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
						validation_steps=max(1,num_val//batch_size),
						epochs=50,
						initial_epoch=0,
						callbacks=[checkpoint_period1, reduce_lr])
	model.save_weights(logdir+"last1.h5")










