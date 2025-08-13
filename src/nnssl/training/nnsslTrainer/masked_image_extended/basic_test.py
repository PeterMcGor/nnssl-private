from BaseMAETrainerExtended import BaseMAETrainerExtended


def unit_test_hook_and_loss():
    import torch
    from types import SimpleNamespace

    # ---- Minimal fake plan/config so build_architecture_and_adaptation_plan works ----
    # If your AbstractBaseTrainer needs more, adapt accordingly.
    dummy_cfg = SimpleNamespace(
        patch_size=(160, 160, 160),
        use_mask_for_norm=[False],
        batch_size=1,
    )
    # Minimal Plan with a dict-like .configurations
    plan = SimpleNamespace(configurations={"default": dummy_cfg})

    trainer = BaseMAETrainerExtended(
        plan=plan,
        configuration_name="default",
        fold=0,
        pretrain_json={},               # not used in this unit test
        device=torch.device("cpu"),     # keep it simple
    )

    # Monkey-patch what AbstractBaseTrainer might expect (if needed)
    trainer.num_input_channels = 1
    trainer.num_output_channels = 1
    trainer.recommended_downstream_patchsize = (160, 160, 160)

    # Build net & loss without full AbstractBaseTrainer init:
    net, adapt = trainer.build_architecture_and_adaptation_plan(
        config_plan=trainer.config_plan, num_input_channels=1, num_output_channels=1
    )
    trainer.network = net
    trainer.loss = trainer.build_loss()
    trainer.device = torch.device("cpu")
    trainer.network.to(trainer.device)
    trainer.batch_size = 1

    # Register hook manually for the unit test
    trainer._register_bottleneck_hook()

    # Fake one batch
    x = torch.randn(1, 1, 160, 160, 160)
    out = trainer.network(x)  # hook should fire

    assert trainer._bottleneck_feat is not None, "Hook did not capture features."
    print("Captured feature shape:", tuple(trainer._bottleneck_feat.shape))

    # Call loss & backward
    loss = trainer.loss(trainer._bottleneck_feat)
    print("Loss (unit):", float(loss))
    loss.backward()
    print("Backward OK")


def main():
    unit_test_hook_and_loss()


if __name__ == "__main()__":
    main()