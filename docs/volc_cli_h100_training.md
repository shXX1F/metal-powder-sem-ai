# Volcengine CLI H100 training workflow

This project should use CLI submission plus TOS-mounted data.

## 1. Install CLI

Run this in Git Bash, WSL, Linux, or a cloud development machine:

```bash
sh -c "$(curl -fsSL https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/cli-binary/install.sh)" && export PATH=$HOME/.volc/bin:$PATH
```

Then configure credentials:

```bash
volc configure
```

Use:

```text
region: cn-beijing
```

Check:

```bash
volc version
ls $HOME/.volc/
```

## 2. Put large files in TOS

Do not upload the `.pth` through the web source-code uploader.

Recommended TOS layout:

```text
tos://tos-mlp-zgci/weights/maskrcnn_particle_last.pth
tos://tos-mlp-zgci/data/GH3536-SA0120724061801-H1/
tos://tos-mlp-zgci/data/boost_round1_gh_hx_full_context/
tos://tos-mlp-zgci/data/fixed_eval_20/
```

The training command in `volc_h100_boost_full_context.yaml` expects these paths after mount:

```text
/tos-mlp-zgci/weights/maskrcnn_particle_last.pth
/tos-mlp-zgci/data/boost_round1_gh_hx_full_context/annotations.json
/tos-mlp-zgci/data/GH3536-SA0120724061801-H1/
```

## 3. Edit the YAML

Open:

```text
volc_h100_boost_full_context.yaml
```

Replace:

```text
REPLACE_WITH_YOUR_QUEUE_NAME
REPLACE_WITH_H100_FLAVOR
```

If the TOS mount reports a prefix error, change:

```text
Prefix: "/"
```

to:

```text
Prefix: ""
```

## 4. Submit task

From the project root:

```bash
volc ml_task submit --conf volc_h100_boost_full_context.yaml
```

If you want to force a task name from CLI:

```bash
volc ml_task submit --conf volc_h100_boost_full_context.yaml --task_name sem-maskrcnn-boost-full-h100
```

## 5. Check task

```bash
volc ml_task list
```

View details:

```bash
volc ml_task get --id TASK_ID
```

View logs:

```bash
volc ml_task log --task TASK_ID --instance worker_0
```

## 6. Output

The trained model will be written to:

```text
tos://tos-mlp-zgci/runs/train_maskrcnn_boost_round1_full_h100/maskrcnn_particle_last.pth
```
