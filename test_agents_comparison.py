#!/usr/bin/env python3
"""
Agent 对比测试脚本
用于验证 Tool Calling Agent 和 ReAct Agent 的功能差异
"""

import asyncio
import json
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.service.langchain_agent import main_graph_execution as tool_calling_execution
from src.service.langchain_react_agent import main_graph_execution as react_execution


async def test_agent(agent_name: str, execution_func, user_input: str, session_id: str):
    """测试单个 Agent"""
    print(f"\n{'='*60}")
    print(f"🤖 测试 {agent_name}")
    print(f"{'='*60}")
    print(f"📝 用户输入: {user_input}")
    print(f"🆔 会话ID: {session_id}")
    print(f"{'─'*60}")
    
    try:
        event_count = 0
        async for event in execution_func(user_input, session_id):
            event_type = event.get("type")
            event_data = event.get("data")
            event_count += 1
            
            # 根据事件类型显示不同的图标和格式
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
                print(f"🔧 【工具调用】: {tool_name} | 参数: {json.dumps(tool_args, ensure_ascii=False)[:100]}... | ID: {call_id}")
            elif event_type == "tool_result":
                tool_name = event_data.get('name', '未知工具')
                content = event_data.get('content', '无内容')[:200]
                call_id = event_data.get('tool_call_id', '无ID')
                print(f"📋 【工具结果】: {tool_name} | 内容: {content}... | ID: {call_id}")
            elif event_type == "chunk":
                print(f"💬 {event_data}", end="", flush=True)
            elif event_type == "stream_end":
                print(f"\n✅ 【流结束】 - 共处理 {event_count} 个事件")
            elif event_type == "error":
                print(f"❌ 【错误】: {event_data}")
            else:
                # 其他未知事件类型
                print(f"❓ 【{event_type}】: {str(event_data)[:100]}...")
                
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


async def compare_agents():
    """对比两种 Agent 的表现"""
    # 测试用例
    test_cases = [
        {
            "input": "你好，请帮我搜索一下最新的人工智能发展趋势",
            "session_id": "test_comparison_001",
            "description": "简单搜索任务"
        },
        {
            "input": "我想了解RAG技术，请先从知识库搜索相关资料，然后结合最新信息给我详细介绍",
            "session_id": "test_comparison_002", 
            "description": "复合任务：知识库检索 + 网络搜索"
        }
    ]
    
    print("🚀 开始 Agent 对比测试")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {test_case['description']}")
        
        # 测试 Tool Calling Agent
        await test_agent(
            "Tool Calling Agent", 
            tool_calling_execution, 
            test_case["input"], 
            f"{test_case['session_id']}_tool_calling"
        )
        
        print("\n" + "─" * 80)
        
        # 测试 ReAct Agent  
        await test_agent(
            "ReAct Agent", 
            react_execution, 
            test_case["input"], 
            f"{test_case['session_id']}_react"
        )
        
        if i < len(test_cases):
            print("\n" + "🔄 准备下一个测试用例..." + "\n")
            await asyncio.sleep(1)  # 短暂暂停
    
    print("\n" + "=" * 80)
    print("🎉 所有测试完成！")
    
    print("\n📊 对比总结:")
    print("┌─────────────────────┬─────────────────────────────────────────────┐")
    print("│      Agent 类型     │                   特点                      │")
    print("├─────────────────────┼─────────────────────────────────────────────┤")
    print("│  Tool Calling Agent │ • 原生工具调用支持                          │")
    print("│                     │ • 更简洁的对话模式                          │")
    print("│                     │ • 适合快速响应场景                          │")
    print("│                     │ • 事件类型: chunk, tool_call, tool_result   │")
    print("├─────────────────────┼─────────────────────────────────────────────┤")
    print("│    ReAct Agent      │ • 推理-行动模式 (Reasoning and Acting)     │")
    print("│                     │ • 显示详细思考过程                          │")
    print("│                     │ • 适合复杂推理场景                          │")
    print("│                     │ • 事件类型: thought, action, observation    │")
    print("└─────────────────────┴─────────────────────────────────────────────┘")


async def quick_test():
    """快速测试单个 Agent"""
    print("🚀 快速测试 ReAct Agent")
    
    await test_agent(
        "ReAct Agent (快速测试)", 
        react_execution, 
        "你好，请简单介绍一下你自己的能力", 
        "quick_test_session"
    )


if __name__ == "__main__":
    print("Agent 对比测试工具")
    print("选择测试模式:")
    print("1. 完整对比测试 (推荐)")
    print("2. 快速测试 ReAct Agent")
    print("3. 退出")
    
    try:
        choice = input("\n请输入选项 (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(compare_agents())
        elif choice == "2":
            asyncio.run(quick_test())
        elif choice == "3":
            print("👋 再见！")
        else:
            print("❌ 无效选项")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"❌ 运行测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
