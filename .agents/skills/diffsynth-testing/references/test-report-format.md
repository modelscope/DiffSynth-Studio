# 测试报告格式

```markdown
# {Series Name} 推理测试报告

## 测试环境
- 设备：{device}
- PyTorch 版本：{version}
- 测试时间：{timestamp}

## 推理测试

| 脚本 | 状态 | 耗时 | 重试 | 输出文件 | 备注 |
|------|------|------|------|---------|------|
| {script_1}.py | ✅ 成功 | 45s | 0 | [output.jpg](link) (2.3MB) | ... |
| {script_2}.py | ❌ 失败 | 30s | 1 (SKIP_DOWNLOAD) | - | 错误信息：... |
| {script_3}.py | ✅ 成功 | 60s | 1 (SKIP_DOWNLOAD) | [output.jpg](link) | 下载失败后重试成功 |

## 失败脚本详情

### {failed_script_name}.py

**退出码**: {exit_code}

**报错信息**:
```
{stderr_tail}
```

**建议修复方案**:
- {suggestion_1}
- {suggestion_2}

**完整日志**: `{error_log_path}`

## 需要人工确认的项目

- [ ] 推理输出 {script_1}.py 的内容是否符合预期（查看 `{output_path}`）
- [ ] 推理输出 {script_3}.py 的内容是否符合预期

## 总结

- 推理脚本：{success}/{total} 成功
- 失败脚本：{error} 个
- 下载重试：{retry_count} 个
- 待人工确认：{count} 项
```

### 输出文件类型

根据模型类型，输出文件格式不同：

| 模型类型 | 输出格式 | 验证方式 |
|---------|---------|---------|
| 图像生成 | `.jpg` / `.png` | 文件存在 + 可打开 |
| 视频生成 | `.mp4` / `.gif` | 文件存在 + 可播放 |
| 音频生成 | `.wav` / `.mp3` | 文件存在 + 可播放 |
| 3D 生成 | `.obj` / `.glb` | 文件存在 + 大小 > 0 |
