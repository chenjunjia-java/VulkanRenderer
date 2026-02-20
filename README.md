## VulkanLearning (clangd / clang-cl / CMake)

### 依赖

- **Vulkan SDK**（需要环境变量 `VULKAN_SDK`）
- **LLVM/Clang**（包含 `clang-cl`）
- **CMake**
- **Ninja**
- **MSVC + Windows SDK**（不用 VS IDE 也行，但需要其头文件/库；脚本会自动加载环境）

### 生成 compile_commands.json（给 clangd 用）

在项目根目录执行：

```powershell
.\scripts\run_with_msvc_env.ps1 -Command "cmake --preset clang-cl-debug"
```

生成位置：

- `build\clang-cl-debug\compile_commands.json`

### 编译（Debug）

```powershell
.\scripts\run_with_msvc_env.ps1 -Command "cmake --build --preset clang-cl-debug"
```

生成的可执行文件：

- `build\clang-cl-debug\VulkanLearning.exe`
