# write a MNISTDataset class which inherits from pytorch dataset. 
# Must be Implemented __getitem__(idx) function which returns 
# image in processed_format(in torch.tensor format and values in [0,1] range)
# Also would be great if we could process raw data into let's say NPZ(np.savez())/zstandard 
# format and then use it while training