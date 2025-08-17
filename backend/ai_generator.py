import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For searching specific course content or detailed educational materials
2. **get_course_outline**: For getting course outlines, structure, and lesson lists

Tool Usage Guidelines:
- **Course outline/structure queries**: Use get_course_outline tool to retrieve course title, course link, and complete lesson list
- **Content-specific questions**: Use search_course_content tool for detailed course materials
- **Multi-round tool calling**: You may use tools in multiple rounds to gather comprehensive information
  - First round: Use tools to gather initial information
  - Second round: Use additional tools if you need more information based on first round results
  - Maximum 2 rounds of tool usage per query
- **Complex queries**: For comparisons, multi-part questions, or cross-course analysis, use multiple tool calls across rounds
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use get_course_outline tool first, then answer
- **Course-specific content questions**: Use search_course_content tool first, then answer
- **Complex analysis questions**: Use multiple tool rounds to gather comprehensive information before answering
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

For course outline responses, always include:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def _make_api_call(self, messages: List, system_content: str, tools: Optional[List] = None):
        """
        Make API call to Claude with consistent error handling.
        
        Args:
            messages: List of conversation messages
            system_content: System prompt content
            tools: Optional tools to include in API call
            
        Returns:
            Anthropic response object
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        return self.client.messages.create(**api_params)
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 rounds of sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        print(f"DEBUG: Starting generate_response with query: {query[:100]}...")
        print(f"DEBUG: Tools available: {len(tools) if tools else 0}")
        print(f"DEBUG: Tool manager: {tool_manager is not None}")
        
        MAX_ROUNDS = 2
        
        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Main loop for up to MAX_ROUNDS rounds
        for round_num in range(1, MAX_ROUNDS + 1):
            # Make API call - include tools only if available
            current_tools = tools if tools else None
            response = self._make_api_call(messages, system_content, current_tools)
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                if not tool_manager:
                    # No tool manager available - can't execute tools
                    return ""
                    
                if round_num < MAX_ROUNDS:
                    # Execute tools and update conversation state
                    should_continue, messages, error = self._handle_tool_execution(
                        response, messages, tool_manager
                    )
                    
                    # If tool execution failed or we shouldn't continue, break to final call
                    if error or not should_continue:
                        break
                        
                    # Continue to next round with updated conversation context
                    continue
                else:
                    # We've reached max rounds but Claude still wants tools - execute them and make final call
                    should_continue, messages, error = self._handle_tool_execution(
                        response, messages, tool_manager
                    )
                    # Since we're at max rounds, make the final call immediately
                    final_response = self._make_api_call(messages, system_content, tools=None)
                    if final_response.content and len(final_response.content) > 0:
                        content_block = final_response.content[0]
                        if hasattr(content_block, 'text'):
                            result = content_block.text
                            # Clean function calls if present
                            if '<function_calls>' in result and '</function_calls>' in result:
                                import re
                                pattern = r'</function_result>\s*(.*?)$'
                                match = re.search(pattern, result, re.DOTALL)
                                if match:
                                    cleaned_result = match.group(1).strip()
                                    if cleaned_result:
                                        return cleaned_result
                            return result
                    return ""
            else:
                # No tool use - return response directly
                if response.content and len(response.content) > 0:
                    # For text responses, content[0] should have text attribute
                    content_block = response.content[0]
                    if hasattr(content_block, 'text'):
                        return content_block.text
                    else:
                        return ""
                else:
                    return ""
        
        # Make final call without tools (for error cases or early breaks)
        final_response = self._make_api_call(messages, system_content, tools=None)
        if final_response.content and len(final_response.content) > 0:
            content_block = final_response.content[0]
            if hasattr(content_block, 'text'):
                result = content_block.text
                print(f"DEBUG: Final response length: {len(result)}, content: {result[:500]}...")
                # Debug: Check if result contains function calls that should be cleaned
                if '<function_calls>' in result and '</function_calls>' in result:
                    print("DEBUG: Found function calls in response, cleaning...")
                    # Extract just the final answer after the function calls
                    import re
                    # Find the last function_result block and get text after it
                    pattern = r'</function_result>\s*(.*?)$'
                    match = re.search(pattern, result, re.DOTALL)
                    if match:
                        cleaned_result = match.group(1).strip()
                        print(f"DEBUG: Cleaned result length: {len(cleaned_result)}")
                        if cleaned_result:
                            return cleaned_result
                print(f"DEBUG: Returning original result length: {len(result)}")
                return result
            else:
                return ""
        else:
            return ""
    
    def _handle_tool_execution(self, initial_response, messages: List, tool_manager):
        """
        Handle execution of tool calls and update conversation state.
        
        Args:
            initial_response: The response containing tool use requests
            messages: Current message list to update
            tool_manager: Manager to execute tools
            
        Returns:
            Tuple of (should_continue: bool, updated_messages: List, error: Optional[str])
        """
        # Add AI's tool use response to messages
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        execution_error = None
        
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    error_message = f"Tool execution error: {str(e)}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": error_message
                    })
                    execution_error = error_message
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Return continuation state
        should_continue = execution_error is None and len(tool_results) > 0
        return should_continue, messages, execution_error