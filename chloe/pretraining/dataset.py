from torch.utils.data import Dataset


class SimPaDataset(Dataset):
    """Class representing Dataset defined from environment for pretraining purposes.
    """

    def __init__(self, env, corrupt_data_flag=True):
        """Init method of dataset.

        Parameters
        ----------
        env: PatientInteractionSimulator
            the environment from which the dataset is derived.
        corrupt_data_flag: boolean
            flag indicating whether or not to corrupt the data to be retrieved.
            By corrupting data, randomly partial data will be retrieved from the
            simulator. Default: True
        """
        self.env = env
        self.corrupt_data_flag = corrupt_data_flag

    def __getitem__(self, idx):
        """Get the item at the provided index.

        Parameters
        ----------
        idx: int
            the index of interest.
        """
        return self.env.get_data_at_index(idx, self.corrupt_data_flag)

    def __len__(self):
        """Get the len of the dataset.
        """
        return self.env.rb.num_rows
