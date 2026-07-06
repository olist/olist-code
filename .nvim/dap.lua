local dap = require("dap")
local dap_python = require("dap-python")
dap_python.setup()

local function is_class_definition(line_content)
	return string.match(line_content, "^%s*class%s+([%w_]+)")
end

local function function_has_self_parameter(line_content)
	local has_self = string.match(line_content, "def%s+[%w_]+%s*%(self[,%s)]")
	return has_self ~= nil
end

local function get_current_test_info()
	local current_line = vim.fn.line(".")
	local current_file = vim.fn.expand("%:p")
	local test_name = nil
	local class_name = nil
	local is_method = false

	for line_num = current_line, 1, -1 do
		local line_content = vim.fn.getline(line_num)
		local next_line_content = vim.fn.getline(line_num + 1)

		local test_match = string.match(line_content, "^%s*def%s+(test_[%w_]+)")
		if test_match then
			test_name = test_match
			is_method = function_has_self_parameter(line_content) or string.match(next_line_content, "self")
			break
		end

		local async_test_match = string.match(line_content, "^%s*async%s+def%s+(test_[%w_]+)")
		if async_test_match then
			test_name = async_test_match
			is_method = function_has_self_parameter(line_content)
			break
		end
	end

	if test_name and is_method then
		for line_num = current_line, 1, -1 do
			local line_content = vim.fn.getline(line_num)
			local class_match = is_class_definition(line_content)
			if class_match then
				class_name = class_match
				break
			end
		end
	end

	return {
		test_name = test_name,
		class_name = class_name,
		is_method = is_method,
	}
end

local function build_test_specifier(test_info)
	if not test_info.test_name then
		return nil
	end

	if test_info.is_method and test_info.class_name then
		return "${file}::" .. test_info.class_name .. "::" .. test_info.test_name
	else
		return "${file}::" .. test_info.test_name
	end
end

dap.configurations.python = {
	{
		type = "python",
		request = "launch",
		name = "Server: Run ASGI (Granian Direct)",
		module = "granian",
		args = {
			"olist_code.server:app",
			"--host",
			"0.0.0.0",
			"--port",
			"3080",
			"--interface",
			"asgi",
		},
		cwd = "${workspaceFolder}",
		console = "integratedTerminal",
		justMyCode = false,
	},
	{
		type = "python",
		request = "launch",
		name = "CLI: olist-code-adapter run",
		module = "olist_code.cli",
		args = { "run" },
		cwd = "${workspaceFolder}",
		console = "integratedTerminal",
		justMyCode = false,
	},
	{
		type = "python",
		request = "launch",
		name = "Pytest: Current Test",
		module = "pytest",
		args = function()
			local test_name = vim.fn.input("Test name: ")
			return {
				"${file}",
				"-k",
				test_name,
				"-vv",
				"-s",
				"--tb=short",
			}
		end,
		console = "integratedTerminal",
		justMyCode = false,
		cwd = "${workspaceFolder}",
		env = {
			PYTHONPATH = "${workspaceFolder}",
		},
	},
	{
		type = "python",
		request = "launch",
		name = "Pytest: Current Test (Auto)",
		module = "pytest",
		args = function()
			local test_info = get_current_test_info()
			local test_specifier = build_test_specifier(test_info)

			if test_specifier then
				return {
					test_specifier,
					"-vv",
					"-s",
					"--tb=short",
				}
			else
				return {
					"${file}",
					"-vv",
					"-s",
					"--tb=short",
				}
			end
		end,
		console = "integratedTerminal",
		justMyCode = false,
		cwd = "${workspaceFolder}",
		env = {
			PYTHONPATH = "${workspaceFolder}",
		},
	},
	{
		type = "python",
		request = "launch",
		name = "Pytest: Current Test Class",
		module = "pytest",
		args = function()
			local test_info = get_current_test_info()

			if test_info.class_name then
				local class_specifier = "${file}::" .. test_info.class_name

				return {
					class_specifier,
					"-vv",
					"-s",
					"--tb=short",
				}
			else
				return {
					"${file}",
					"-vv",
					"-s",
					"--tb=short",
				}
			end
		end,
		console = "integratedTerminal",
		justMyCode = false,
		cwd = "${workspaceFolder}",
		env = {
			PYTHONPATH = "${workspaceFolder}",
		},
	},
}

local function debug_current_test()
	local test_info = get_current_test_info()
	print("=== Informações do Teste ===")
	print("Nome do teste: " .. (test_info.test_name or "Não encontrado"))
	print("Nome da classe: " .. (test_info.class_name or "N/A"))
	print("É método de classe: " .. tostring(test_info.is_method))

	local specifier = build_test_specifier(test_info)
	print("Especificador pytest: " .. (specifier or "N/A"))
end

vim.api.nvim_create_user_command("DebugTestInfo", debug_current_test, {})
