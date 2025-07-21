from networks.MedNeXt.mednextv1.MedNeXtV1_EffiDec3D import MedNeXt_EffiDec3D

def create_mednextv1_effidec3d_small(num_input_channels, num_classes, kernel_size=3, n_channels=32, ds=False, mode='train'):

    return MedNeXt_EffiDec3D(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        exp_r=2,                         
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2], 
        mode='train'
    )


def create_mednextv1_effidec3d_base(num_input_channels, num_classes, kernel_size=3, n_channels=32, ds=False, mode='train'):

    return MedNeXt_EffiDec3D(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2], 
        mode=mode
    )


def create_mednextv1_effidec3d_medium(num_input_channels, num_classes, kernel_size=3, n_channels=32, ds=False, mode='train'):

    return MedNeXt_EffiDec3D(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3], #[3,4,4,4,4,4,4,4,3]
        checkpoint_style = 'outside_block', 
        mode=mode
    )


def create_mednextv1_effidec3d_large(num_input_channels, num_classes, kernel_size=3, n_channels=32, ds=False, mode='train'):

    return MedNeXt_EffiDec3D(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        exp_r=[3,4,8,8,8,8,8,4,3],                          
        kernel_size=kernel_size,                     
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,8,8,8,8,8,4,3],
        checkpoint_style = 'outside_block',
        mode=mode
    )


def create_mednextv1_effidec3d(num_input_channels, num_classes, model_id, kernel_size=3, n_channels=32,
                      deep_supervision=False, mode='train'):

    model_dict = {
        'S': create_mednextv1_effidec3d_small,
        'B': create_mednextv1_effidec3d_base,
        'M': create_mednextv1_effidec3d_medium,
        'L': create_mednextv1_effidec3d_large,
        }
    
    return model_dict[model_id](
        num_input_channels, num_classes, kernel_size, n_channels, deep_supervision, mode=mode
        )


if __name__ == "__main__":

    model = create_mednextv1_effidec3d_large(1, 3, 3, False)
    print(model)