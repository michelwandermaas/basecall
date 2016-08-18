import h5py
def get_scale_and_shift(file_name, strand, read_type):
    h5 = h5py.File(file_name, "r")
    if(strand == 1):
        base_loc = "Analyses/Basecall_1D_000"
    else:
        base_loc = "Analyses/Basecall_2D_000"
    scale = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["scale"]
    scale_sd = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["scale_sd"]
    shift = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["shift"]
    return scale, scale_sd, shift
def scale(mean,stdv,scale_value,scale_sd,shift):
    mean = float(mean)
    stdv = float(stdv)
    shift = float(shift)
    scale_value = float(scale_value)
    scale_sd = float(scale_sd)
    shift = float(shift)

    mean = (mean - shift) / scale_value
    stdv = stdv / scale_sd
    return mean, stdv