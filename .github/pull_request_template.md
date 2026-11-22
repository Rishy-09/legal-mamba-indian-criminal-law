## Summary

<!-- Briefly describe the change. -->

## Type of Change

- [ ] Bug fix
- [ ] Model / architecture
- [ ] Training
- [ ] Data / tokenizer
- [ ] Evaluation
- [ ] Tests / CI
- [ ] Documentation
- [ ] Reproducibility / cleanup
- [ ] Other

## Components Affected

<!-- Examples: Mamba model, training loop, tokenizer, corpus pipeline, checkpointing -->

## Validation

- [ ] Source compiles successfully
- [ ] Existing automated tests pass
- [ ] New tests were added where appropriate
- [ ] Documentation was updated where required

### Validation Commands

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
```

## Model / Checkpoint Impact

<!--
Describe changes to architecture, configuration, tokenizer, training,
or checkpoint compatibility. Write "None" if unaffected.
-->

## Results

<!-- Add updated loss, evaluation results, plots, or generation examples when applicable. -->

## Additional Notes

<!-- Compatibility information, screenshots, logs, limitations, etc. -->