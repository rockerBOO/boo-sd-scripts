# Plain MOSS-Image LoRA training through boo-sd-scripts. Same CLI as sd-scripts' moss_image_train_network.py, plus the
# --log_grad_metrics* options. This is the seam where extensions get added, and it is used to test the whole pipeline.

import boo_sd_scripts._sd_scripts  # noqa: F401  (must come before the sd-scripts imports)
import library.args as args_util
from boo_sd_scripts.moss_image import grad_metrics
from library import moss_image_train_utils
from moss_image_train_network import MossImageNetworkTrainer
from moss_image_train_network import setup_parser as moss_setup_parser


class BooMossImageNetworkTrainer(MossImageNetworkTrainer):
    def train(self, args):
        grad_metrics.validate_grad_metrics_args(args)
        self._grad_args = args
        self._grad_logs: dict = {}
        super().train(args)

    def all_reduce_network(self, accelerator, network):
        # Called right after backward and before clipping, so the norms are pre-clip.
        super().all_reduce_network(accelerator, network)
        args = self._grad_args
        if not args.log_grad_metrics or len(accelerator.trackers) == 0:
            return
        self._grad_logs = grad_metrics.collect_grad_metrics(network.parameters())
        if args.log_grad_metrics_per_module or args.log_grad_metrics_block_regex is not None:
            self._grad_logs.update(
                grad_metrics.collect_grad_metrics_by_module(
                    accelerator.unwrap_model(network).named_parameters(),
                    args.log_grad_metrics_block_regex,
                    per_module=args.log_grad_metrics_per_module,
                )
            )

    def generate_step_logs(self, *args, **kwargs):
        logs = super().generate_step_logs(*args, **kwargs)
        logs.update(self._grad_logs)
        self._grad_logs = {}
        return logs


def setup_parser():
    parser = moss_setup_parser()
    grad_metrics.add_grad_metrics_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.show_timesteps:
        moss_image_train_utils.show_timesteps(args)
    else:
        trainer = BooMossImageNetworkTrainer()
        trainer.train(args)
