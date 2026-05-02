1. 启动echo服务器
2. 启动openai privacy服务器
3. 启动rapi服务器
4. 命令ai进行手动测试
<info>
(openai-privacy) ➜  rapi git:(master) ✗ just start-echo-server
=== Starting Echo Server on port 18081 ===

[1/2] Checking prerequisites...
  ✓ cargo found: cargo 1.94.0 (85eff7c80 2026-01-15)
[2/2] Starting Echo Server...
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/echo_server`
Echo server listening on 0.0.0.0:18081
warning: `rapi` (bin "rapi") generated 5 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/rapi`
Transparent forwarder listening on http://0.0.0.0:13000
</info>
<info>
echo服务器和rapi服务器都已经启动
</info>
<target>
1. 对rapi进行测试，不仅需要覆盖正常的功能测试，还需要边界测试，错误测试，性能测试。
</target>

<instruct>
1. 对rapi的功能进行手动测试。
</instruct>
