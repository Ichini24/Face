from keras.callbacks import ModelCheckpoint


import data, model


dataproc = data.dataProcess(299, 299)
dataproc.create_train_data()
imgs, mask = dataproc.load_train_data()

model = model.get_model()

model_checkpoint = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0, period=50)
print('Fitting model...')
model.fit(imgs, mask, batch_size=4, epochs=10000, verbose=1, validation_split=0.2,
		  shuffle=True, callbacks=[model_checkpoint])
