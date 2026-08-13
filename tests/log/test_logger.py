from ml_commons.log.logger import NullLogger, WandBLogger


def test_null_logger_all_methods_noop():
    logger = NullLogger()
    logger.add_elements({"a": 1})
    logger.set_element_step_metric({"a": "step"})
    logger.set_log_data({"a": 2})
    logger.sum_log_data({"a": 3})
    logger.log_data("a")
    logger.set_prefix({"a": "prefix-"})
    logger.reset("a")
    logger.finish()


def test_wandb_logger_init_calls_wandb(mock_wandb, sample_run_info):
    mock_init, mock_run = mock_wandb

    WandBLogger(sample_run_info, entity="me", project="proj",
               hyperparameters={"lr": 0.1}, elements={"loss": 0.0})

    mock_init.assert_called_once_with(
        entity="me", project="proj",
        name=sample_run_info.run_name(),
        tags=sample_run_info.tags(),
        job_type="train",
        config={"lr": 0.1},
        group=sample_run_info.group(),
    )
    mock_run.define_metric.assert_called_once_with("*", step_metric="global_step")


def test_wandb_logger_finish(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={})
    logger.finish()
    mock_run.finish.assert_called_once()


def test_wandb_logger_add_elements_updates_both_current_and_start(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"a": 0})

    logger.add_elements({"b": 10})
    logger.sum_log_data({"b": 5})
    logger.reset("b")
    logger.log_data()

    assert mock_run.log.call_args.kwargs["data"] == {"a": 0, "b": 10}


def test_wandb_logger_sum_log_data_accumulates(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"loss": 0.0})

    logger.sum_log_data({"loss": 1.0})
    logger.sum_log_data({"loss": 2.0})
    logger.log_data()

    assert mock_run.log.call_args.kwargs["data"] == {"loss": 3.0}


def test_wandb_logger_reset_specific_field_only(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"a": 0, "b": 0})

    logger.sum_log_data({"a": 5, "b": 5})
    logger.reset("a")
    logger.log_data()

    assert mock_run.log.call_args.kwargs["data"] == {"a": 0, "b": 5}


def test_wandb_logger_reset_all_when_no_fields_given(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"a": 0, "b": 0})

    logger.sum_log_data({"a": 5, "b": 5})
    logger.reset()
    logger.log_data()

    assert mock_run.log.call_args.kwargs["data"] == {"a": 0, "b": 0}


def test_wandb_logger_set_element_step_metric(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"loss": 0.0})

    logger.set_element_step_metric({"loss": "epoch"})

    mock_run.define_metric.assert_called_with("loss", step_metric="epoch")


def test_wandb_logger_log_data_no_filter_includes_prefixed_key(mock_wandb, sample_run_info):
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"loss": 0.0, "epoch": 0})

    logger.set_prefix({"loss": "0-"})
    logger.set_log_data({"loss": 1.5, "epoch": 3})
    logger.log_data()

    assert mock_run.log.call_args.kwargs["data"] == {"0-loss": 1.5, "epoch": 3}


def test_wandb_logger_log_data_filter_matches_against_prefixed_key(mock_wandb, sample_run_info):
    # log_data() applies the prefix before filtering, so a prefixed field must be
    # requested by its *prefixed* name to survive the filter — its bare name won't match.
    _, mock_run = mock_wandb
    logger = WandBLogger(sample_run_info, entity="me", project="proj",
                         hyperparameters={}, elements={"loss": 0.0, "epoch": 0})

    logger.set_prefix({"loss": "0-"})
    logger.set_log_data({"loss": 1.5, "epoch": 3})

    logger.log_data("loss")
    assert mock_run.log.call_args.kwargs["data"] == {}

    logger.log_data("0-loss")
    assert mock_run.log.call_args.kwargs["data"] == {"0-loss": 1.5}
