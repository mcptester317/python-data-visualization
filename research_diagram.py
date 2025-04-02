import graphviz

def create_research_proposal_diagram():
    """
    Create a directed graph visualization for a research proposal framework.
    """
    # Create a new directed graph
    dot = graphviz.Digraph(format='png', comment='Research Proposal Framework')

    # Configure graph attributes for better readability
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', fontsize='10', height='0.7')

    # Research Problem Definition
    dot.node('A', 'Research Problem\n(Define research question)', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightblue')

    # Literature Review
    dot.node('B', 'Literature Review\n(Analyze existing work)', 
            shape='rectangle', 
            style='filled', 
            fillcolor='lightgray')

    # Methodology Branches
    dot.node('C1', 'Data Collection\n(Methods & Tools)', 
            shape='diamond', 
            style='filled', 
            fillcolor='lightgreen')
    dot.node('C2', 'Data Analysis\n(Techniques & Approaches)', 
            shape='diamond', 
            style='filled', 
            fillcolor='lightgreen')
    dot.node('C3', 'Validation\n(Testing & Verification)', 
            shape='diamond', 
            style='filled', 
            fillcolor='lightgreen')

    # Expected Outcomes
    dot.node('D', 'Expected Outcomes\n(Research Contributions)', 
            shape='rectangle', 
            style='filled', 
            fillcolor='gold')

    # Timeline & Resources
    dot.node('E', 'Timeline & Resources\n(Project Planning)', 
            shape='rectangle', 
            style='filled', 
            fillcolor='lightpink')

    # Connect the nodes
    dot.edge('A', 'B', 'Informs')
    dot.edge('B', 'C1', 'Guides')
    dot.edge('B', 'C2', 'Directs')
    dot.edge('B', 'C3', 'Shapes')
    dot.edge('C1', 'D', 'Contributes to')
    dot.edge('C2', 'D', 'Leads to')
    dot.edge('C3', 'D', 'Validates')
    dot.edge('D', 'E', 'Plans')

    # Return the graph
    return dot

def create_prompt_analysis_diagram():
    """
    Create a directed graph visualization for prompt analysis and safety framework.
    """
    # Create a new directed graph
    dot = graphviz.Digraph(format='png', comment='Prompt Safety Analysis Framework')

    # Configure graph attributes for better readability
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', fontsize='10', height='0.7')

    # Input Prompt
    dot.node('A', 'Input Prompt\n(User Request)', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightblue')

    # Intent Analysis
    dot.node('B', 'Intent Analysis\n(Understanding True Purpose)', 
            shape='diamond', 
            style='filled', 
            fillcolor='lightgray')

    # Path Planning
    dot.node('C', 'Path Planning\n(Generate Multiple Approaches)', 
            shape='rectangle', 
            style='filled', 
            fillcolor='lightgreen')

    # Safety Evaluation
    dot.node('D1', 'Path 1\n(Safe Approach)', 
            shape='ellipse', 
            style='filled', 
            fillcolor='palegreen')
    dot.node('D2', 'Path 2\n(Moderate Risk)', 
            shape='ellipse', 
            style='filled', 
            fillcolor='khaki')
    dot.node('D3', 'Path 3\n(High Risk)', 
            shape='ellipse', 
            style='filled', 
            fillcolor='lightcoral')

    # Final Decision
    dot.node('E', 'Safety Decision\n(Choose Safest Path)', 
            shape='rectangle', 
            style='filled', 
            fillcolor='lightblue')

    # Response Generation
    dot.node('F', 'Generate Response\n(Execute Safe Path)', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightgreen')

    # Connect the nodes
    dot.edge('A', 'B', 'Analyze')
    dot.edge('B', 'C', 'Plan')
    dot.edge('C', 'D1', 'Option 1')
    dot.edge('C', 'D2', 'Option 2')
    dot.edge('C', 'D3', 'Option 3')
    dot.edge('D1', 'E', 'Evaluate')
    dot.edge('D2', 'E', 'Evaluate')
    dot.edge('D3', 'E', 'Evaluate')
    dot.edge('E', 'F', 'Execute')

    # Return the graph
    return dot

def create_ai_defense_framework():
    """
    Create a directed graph visualization for an AI defense framework.
    """
    # Create a new directed graph
    dot = graphviz.Digraph(format='png')

    # Intent Extraction
    dot.node('A', 'Intent Extraction\n(Identify true user intent)', 
            shape='parallelogram', style="filled", fillcolor="lightblue")

    # Reasoning Tree Generation
    dot.node('B', 'Reasoning Tree Generation\n(Construct reasoning paths)', 
            shape='diamond', style="filled", fillcolor="lightgray")

    # Outcome Evaluation (Tree Paths)
    dot.node('C1', 'Path 1: Safe Intent\n✅', 
            shape='ellipse', style="filled", fillcolor="lightgreen")
    dot.node('C2', 'Path 2: Potentially Unsafe Intent\n⚠️', 
            shape='ellipse', style="filled", fillcolor="gold")
    dot.node('C3', 'Path 3: Adversarial Manipulation\n❌', 
            shape='ellipse', style="filled", fillcolor="red")

    # Memory Sanitization
    dot.node('D', 'Memory Sanitization\n(Forget unsafe prompts)', 
            shape='parallelogram', style="filled", fillcolor="lightblue")

    # Edges for Flow
    dot.edge('A', 'B', label='Generate possible reasoning paths')
    dot.edge('B', 'C1', label='Interpretation 1')
    dot.edge('B', 'C2', label='Interpretation 2')
    dot.edge('B', 'C3', label='Interpretation 3')

    # Evaluation & Decision
    dot.edge('C1', 'D', label='Choose safest interpretation')
    dot.edge('C2', 'D', label='Sanitize uncertain inputs')
    dot.edge('C3', 'D', label='Reject manipulation')

    # Return the graph
    return dot

def main():
    """Generate all diagrams and save them as image files."""
    # Create research proposal diagram
    research_diagram = create_research_proposal_diagram()
    research_diagram.render('research_proposal_framework', format='png', cleanup=True)
    print("Research proposal diagram created as 'research_proposal_framework.png'")
    
    # Create prompt analysis diagram
    prompt_diagram = create_prompt_analysis_diagram()
    prompt_diagram.render('prompt_safety_framework', format='png', cleanup=True)
    print("Prompt safety diagram created as 'prompt_safety_framework.png'")
    
    # Create AI defense framework diagram
    defense_diagram = create_ai_defense_framework()
    defense_diagram.render('ai_defense_framework', format='png', cleanup=True)
    print("AI defense framework diagram created as 'ai_defense_framework.png'")

if __name__ == "__main__":
    main()