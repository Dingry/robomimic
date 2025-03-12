from robomimic.scripts.config_gen.dc_config_gen_utils import *



def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    ### Define dataset variants to train on ###
    generator.add_param(
        key="train.data",
        name="newds",
        group=123456,
        values_and_names=[
            # (get_robocasa_ds("dc24", src="human", eval=["PnP5FromCuttingboardToPanNoDistractorSplitA"], filter_key="50_demos"), "human-50"), # training on human datasets
            (get_robocasa_ds("dc24", src="mg", eval=["PosttrainPnPNovelFromPlateToPanSplitA"], filter_key="100_demos"), "24dc-mg-100_true"), # training on MimicGen datasets
            (get_robocasa_ds("dc24", src="mg", eval=["PosttrainPnPNovelFromPlateToPanSplitA"], filter_key="300_demos"), "24dc-mg-300_true"), # training on MimicGen datasets
            (get_robocasa_ds("dc24", src="mg", eval=["PosttrainPnPNovelFromPlateToPanSplitA"], filter_key="1000_demos"), "24dc-mg-1000_true"), # training on MimicGen datasets

            (get_robocasa_ds("PosttrainPnPNovelFromPlateToPanSplitA", src="mg", filter_key="50_demos"), "1dc-mg-50_true"),
            (get_robocasa_ds("PosttrainPnPNovelFromPlateToPanSplitA", src="mg", filter_key="100_demos"), "1dc-mg-100_true"),
            (get_robocasa_ds("PosttrainPnPNovelFromPlateToPanSplitA", src="mg", filter_key="300_demos"), "1dc-mg-300_true"),
            (get_robocasa_ds("PosttrainPnPNovelFromPlateToPanSplitA", src="mg", filter_key="1000_demos"), "1dc-mg-1000_true"),

            # composite tasks
            (get_robocasa_ds("PnP5FromCuttingboardToPanNoDistractorSplitA", src="mg", filter_key="50_demos"), "pnp5-mg-50-debug"),
            (get_robocasa_ds("PnP5FromCuttingboardToPanNoDistractorSplitA", src="mg", eval=[], filter_key="100_demos"), "pnp5-mg-100"),
            (get_robocasa_ds("PnP5FromCuttingboardToPanNoDistractorSplitA", src="mg", eval=[], filter_key="300_demos"), "pnp5-mg-300"),
            (get_robocasa_ds("PnP5FromCuttingboardToPanNoDistractorSplitA", src="mg", eval=[], filter_key="1000_demos"), "pnp5-mg-1000"),
        ]
    )

    """
    ### Uncomment this code to fine-tune on existing checkpoint ###
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            (None, "none"),
            # ("set checkpoint pth path here", "trained-ckpt"),
        ],
    )
    """

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[get_output_dir(args, algo_dir=algo_name_short)]
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
