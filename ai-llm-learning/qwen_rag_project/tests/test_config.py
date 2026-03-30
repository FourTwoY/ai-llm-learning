from pathlib import Path

import config as config_module


def test_load_config_merges_user_config_with_defaults(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
models:
  generation: "test-gen-model"
retrieval:
  top_k: 8
""",
        encoding="utf-8"
    )

    cfg = config_module.load_config(config_file)

    assert cfg["models"]["generation"] == "test-gen-model"
    assert cfg["retrieval"]["top_k"] == 8

    # 默认配置里的字段应仍然存在
    assert "chunking" in cfg
    assert "paths" in cfg
    assert "embedding" in cfg


def test_get_config_supports_reload(monkeypatch, tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
retrieval:
  top_k: 5
""",
        encoding="utf-8"
    )

    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)

    cfg1 = config_module.get_config(reload=True)
    assert cfg1["retrieval"]["top_k"] == 5

    config_file.write_text(
        """
retrieval:
  top_k: 9
""",
        encoding="utf-8"
    )

    # 不 reload 时，应该还是旧缓存
    cfg2 = config_module.get_config()
    assert cfg2["retrieval"]["top_k"] == 5

    # reload 后，应拿到新配置
    cfg3 = config_module.get_config(reload=True)
    assert cfg3["retrieval"]["top_k"] == 9