#!/usr/bin/env python3
"""
ReAct Agent 修复验证测试脚本
用于验证 ReAct Agent 是否能正确处理 JSON 格式的工具输入
"""

import asyncio
import json
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.service.langchain_react_agent import main_graph_execution


async def test_react_agent_fix():
    """测试 ReAct Agent 的 JSON 输入处理修复"""
    print("🚀 测试 ReAct Agent JSON 输入处理修复")
    print("=" * 60)
    
    # 测试用例：使用地图工具查询普宁
    test_input = "请帮我查询普宁的地理位置信息"
    session_id = "test_react_fix_001"
    
    print(f"📝 测试输入: {test_input}")
    print(f"🆔 会话ID: {session_id}")
    print("─" * 60)
    
    try:
        event_count = 0
        tool_calls = []
        errors = []
        
        async for event in main_graph_execution(test_input, session_id):
            event_type = event.get("type")
            event_data = event.get("data")
            event_count += 1
            
            # 打印事件
            if event_type == "thought":
                print(f"🤔 【思考】: {event_data}")
            elif event_type == "action":
                print(f"🎯 【行动】: {event_data}")
            elif event_type == "observation":
                print(f"👀 【观察】: {event_data}")
            elif event_type == "tool_call":
                tool_name = event_data.get('name', '未知工具')
                tool_args = event_data.get('args', {})
                call_id = event_data.get('id', '无ID')
                tool_calls.append({
                    'name': tool_name,
                    'args': tool_args,
                    'id': call_id
                })
                print(f"🔧 【工具调用】: {tool_name}")
                print(f"   📋 参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
                print(f"   🆔 ID: {call_id}")
            elif event_type == "tool_result":
                tool_name = event_data.get('name', '未知工具')
                content = event_data.get('content', '无内容')
                call_id = event_data.get('tool_call_id', '无ID')
                print(f"📋 【工具结果】: {tool_name}")
                print(f"   ✅ 内容: {content[:200]}{'...' if len(content) > 200 else ''}")
                print(f"   🆔 ID: {call_id}")
            elif event_type == "chunk":
                print(f"💬 {event_data}", end="", flush=True)
            elif event_type == "stream_end":
                print(f"\n✅ 【流结束】 - 共处理 {event_count} 个事件")
            elif event_type == "error":
                error_msg = event_data
                errors.append(error_msg)
                print(f"❌ 【错误】: {error_msg}")
                
        # 测试总结
        print("\n" + "=" * 60)
        print("📊 测试总结:")
        print(f"   🔢 总事件数: {event_count}")
        print(f"   🔧 工具调用数: {len(tool_calls)}")
        print(f"   ❌ 错误数: {len(errors)}")
        
        if errors:
            print("\n⚠️  发现错误:")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
            return False
        else:
            print("\n🎉 测试通过！没有发现 JSON 输入相关错误。")
            return True
            
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_json_parsing():
    """测试 JSON 解析逻辑"""
    print("\n🧪 测试 JSON 解析逻辑")
    print("─" * 30)
    
    test_cases = [
        '{"city": "普宁"}',
        'simple_string',
        '{"query": "测试查询", "limit": 5}',
        'invalid_json{',
    ]
    
    import json
    
    for test_input in test_cases:
        try:
            parsed = json.loads(test_input)
            print(f"✅ '{test_input}' -> {parsed}")
        except json.JSONDecodeError:
            print(f"⚠️  '{test_input}' -> 非JSON格式，将作为字符串处理")


if __name__ == "__main__":
    print("ReAct Agent 修复验证测试工具")
    
    try:
        # 首先测试 JSON 解析逻辑
        asyncio.run(test_json_parsing())
        
        # 然后测试完整的 ReAct Agent
        success = asyncio.run(test_react_agent_fix())
        
        if success:
            print("\n🎊 所有测试通过！ReAct Agent 修复成功。")
            sys.exit(0)
        else:
            print("\n💥 测试失败，需要进一步调试。")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
