from config.config_schema import load_config
from config.validator import ConfigValidator

# 测试配置加载
try:
    config = load_config("examples/sample_config.yaml")
    print("✓ 配置文件加载成功")
    print(f"项目名称: {config.project_name}")
    print(f"数据集类型: {config.dataset.type}")
    print(f"评估指标: {config.metrics}")

    # 测试配置验证
    validator = ConfigValidator()
    is_valid = validator.validate(config)

    report = validator.get_validation_report()
    print(f"✓ 配置验证: {'通过' if is_valid else '失败'}")

    if report["errors"]:
        print("错误:")
        for error in report["errors"]:
            print(f"  - {error}")

    if report["warnings"]:
        print("警告:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

except Exception as e:
    print(f"✗ 配置测试失败: {e}")
    import traceback

    traceback.print_exc()
