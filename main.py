from keras.callbacks import ModelCheckpoint


import data, model


dataproc = data.dataProcess(299, 299)
dataproc.create_train_data()
imgs, mask = dataproc.load_train_data()

model = model.get_model()
#board = tenBoard.on_epoch_end()
#board = TensorBoard(log_dir='logs//', histogram_freq=0,
 #         write_graph=True, write_images=True)
#board = keras.callbacks.TensorBoard(log_dir='logs//', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
model_checkpoint = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0, period=50)
print('Fitting model...')
model.fit(imgs, mask, batch_size=4, epochs=10000, verbose=1, validation_split=0.2,
		  shuffle=True, callbacks=[model_checkpoint])
