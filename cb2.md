

``` python
import asyncio
import nest_asyncio
import os
import inspect
import logging
import csv

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

nest_asyncio.apply()

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="deepseek-r1:1.5b1a",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Read CSV data and build a formatted string
    csv_data = ""
    with open("./42a.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data += f"Topic: {row['Topic']}\n"
            csv_data += f"Key Concepts/Themes: {row['Key Concepts/Themes']}\n\n"

    # Insert the CSV data into the RAG system
    rag.insert(csv_data)

    # Test different query modes with updated questions

    print("\nNaive Search:")
    print(
        rag.query(
            "What are the main topics and their associated themes?",
            param=QueryParam(mode="naive")
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            "Which topic emphasizes the discipline of sensory focus and aesthetic enhancement?",
            param=QueryParam(mode="local")
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            "How do the themes reflect ideas of personal growth and moderation across different topics?",
            param=QueryParam(mode="global")
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            "Can you summarize how each topic combines concepts of self-expression, beauty, and resource management?",
            param=QueryParam(mode="hybrid")
        )
    )

    # Stream response for one of the queries
    resp = rag.query(
        "Can you summarize how each topic combines concepts of self-expression, beauty, and resource management?",
        param=QueryParam(mode="hybrid", stream=True),
    )

    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":
    main()
```

    INFO: Process 65859 Shared-Data already initialized (multiprocess=False)
    INFO:Load (0, 768) data
    INFO:Init {'embedding_dim': 768, 'metric': 'cosine', 'storage_file': './dickens/vdb_entities.json'} 0 data
    INFO:Load (0, 768) data
    INFO:Init {'embedding_dim': 768, 'metric': 'cosine', 'storage_file': './dickens/vdb_relationships.json'} 0 data
    INFO:Load (4, 768) data
    INFO:Init {'embedding_dim': 768, 'metric': 'cosine', 'storage_file': './dickens/vdb_chunks.json'} 4 data
    INFO: Process 65859 storage namespace already initialized: [full_docs]
    INFO: Process 65859 storage namespace already initialized: [text_chunks]
    INFO: Process 65859 storage namespace already initialized: [llm_response_cache]
    INFO: Process 65859 storage namespace already initialized: [doc_status]
    INFO: Process 65859 storage namespace already initialized: [full_docs]
    INFO: Process 65859 storage namespace already initialized: [text_chunks]
    INFO: Process 65859 storage namespace already initialized: [llm_response_cache]
    INFO: Process 65859 storage namespace already initialized: [doc_status]


    Naive Search:
    <think>
    Okay, I need to figure out how to respond to this query about the document chunks provided. The user has asked for "What are the main topics and their associated themes?" So, they want me to identify the key areas and related concepts from each topic chunk.

    First, let me look at each topic and see what they cover. Starting with Topic 22: Everyone has a void—existential suffering, emotional dissatisfaction, universal struggles, empathy, inner fulfillment. The theme here seems to be about feeling a hole or emptiness that's hard to escape but not always worth fighting for.

    Moving on to Topic 23: Down with meditation—any form helpful? It mentions almost every meditation form is useful. This ties into self-improvement and the idea that meditating in different ways can help, so I should highlight various forms of meditation.

    Topic 24 talks about gotta git up to git down—concentration for awareness, productivity, mental discipline. It's about practical strategies for staying focused on both creativity and execution.

    Topic 25: Animal/vegetarian astrology—is all natural? This seems more about the symbolism in nature and spirituality, emphasizing that beauty is found through physical and spiritual aspects.

    Topic 26: Psycho vs socio path—personality issues, social adaptation, behavioral extremes. It's about mental health and how one's behavior can affect others or vice versa.

    Topic 27: Paradox with complexity—apparent paradoxical traits like emotional disruption in complex systems. This is more of a philosophical or theoretical discussion on how interconnectedness leads to certain behaviors without straightforward explanations.

    Topic 28: heaven vs no need to speculate—thinking about divine presence and inner peace, avoiding speculative judgments. It's about personal spirituality and seeking spiritual fulfillment.

    Topic 29: Time—it's not fear but time management with tools like stopwatches. It focuses on how to effectively use time without getting distracted or burnout.

    Topic 30: Feminism—hearing from women's perspectives, challenges faced in their world, social justice issues. It covers cultural perspectives and specific social issues within the feminist community.

    Topic 31: Neech—jews as masters of slaves, examining their mindset. This is about Jewish identity and how they treat others in a particular way based on their beliefs.

    Topic 32: Buddhism—mental aspects discussed here, focusing on thoughts and their effects. It's about the philosophy of Buddhism regarding mind and mental states.

    Topic 33: Dreams— living in a dream state, turning it on or off, developing consciousness in subconscious. This is about sleep and its implications for one's experiences.

    Looking at each topic chunk, I need to identify themes that connect these topics. The main theme seems to revolve around interconnectedness, spirituality, emotional well-being, mental health, and personal identity. Each topic addresses different facets of these broader areas.

    I should structure the response by first identifying all the key themes from the document chunks, then tie them back to the main discussion points like existential suffering, meditation techniques, astrology symbolism, etc., ensuring that each theme is clearly connected.
    </think>

    The document chunks collectively explore a rich tapestry of themes and concepts, with several central ideas emerging from their content:

    ### 1. **Existential sufferance and emotional dissatisfaction**  
       - Topics include the void concept (Topic 22), down with meditation (Topic 23), animal/vegetarian astrology (Topic 25), transcendental paradoxes (Topic 27), and inner fulfillment (Topic 20). These themes all touch on the struggle, suffering, and resilience experienced by individuals in various contexts, reflecting a deeper exploration of existential consciousness.

    ### 2. **Meditation and self-improvement**  
       - Topics like down with meditation (Topic 23) and gotta git up to git down (Topic 24) emphasize the role of meditation as a universal tool for personal growth, concentration, and focus in both mental and physical aspects.

    ### 3. **Nature and symbolism**  
       - The topics discussed include animal/vegetarian astrology (Topic 25), symbolism through nature and spirituality, which explore how perceptions from natural and spiritual realms influence individual experiences and beliefs.

    ### 4. **Complexity and interconnectedness**  
       - Topics like the paradox with complexity (Topic 27) and psycho vs socio paths (Topic 26) address broader questions about interconnectedness, social dynamics, and the ways in which one's actions can affect others and vice versa.

    ### 5. **Spiritual spirituality and inner fulfillment**  
       - Topics such as heaven vs no need to speculate (Topic 28), personal spirituality (Topic 29), and existential void (Topic 22) delve into themes of divine presence, spiritual mindfulness, and the pursuit of inner peace, respectively.

    ### 6. **Personal identity and cultural perspectives**  
       - Topics like the neutrals orientation in Topic 31, Jewish mindset in Topic 31, feminist perspectives in Topic 30, and transcendental recursions (Topic 27) explore various facets of personal identity, societal identity, and cultural practices.

    ### 7. **Time management and well-being**  
       - The theme of time is addressed in Topic 29, focusing on the importance of time management with tools like stopwatches, emphasizing how effective use of time can lead to better outcomes.

    ### 8. **Moral and ethical considerations**  
       - Topics such as Nietzschean balance (Topic 20), ethical boundaries (Topic 19), and universal struggles (Topic 23) explore questions about the moral, ethical, and cultural implications of various aspects of existence.

    In summary, the document chunks collectively address a wide array of themes and concepts, reflecting a comprehensive exploration of existential consciousness, personal identity, spirituality, emotional well-being, mental health, and self-improvement. Each topic touches on different facets of these broader areas, with several central ideas including existential sufferance, meditation's role in self-improvement, symbolism through nature, interconnectedness, spiritual spirituality, and the importance of time management.

    Local Search:
    <think>
    Alright, I need to figure out which topic in the provided knowledge base emphasizes the discipline of sensory focus and aesthetic enhancement. Let me start by reviewing each topic and its description.

    The first topic is "Existential Void." It talks about a hole or void in one's existence where satisfaction isn't easy, leading to emotional discomfort if unsatisfied. So that relates to dissatisfaction but not specifically sensory focus.

    Next is "Meditation Benefits." This section discusses various meditation forms being beneficial for overall well-being and inner peace. While it's related to relaxation and calmness, it doesn't mention anything about focusing on senses or enhancing aesthetics.

    Then there's "Soothing Mindfulness Practices." This seems to cover techniques like deep breathing or meditation focused on the mind. Again, it's more about mental relaxation rather than sensory focus or aesthetic enhancement.

    The topic titled "Discipline of Sensory Focus" stands out. It directly discusses focusing on one's senses and enhancing sensory experience. The key points are how each sense is given due weight and that this discipline can lead to a deeper connection with the world, which aligns perfectly with the theme of enhancing aesthetics through focus.

    Looking further down, "Meditation Benefits in Action" and others relate to meditation techniques but don't specifically delve into sensory focus. The last topic discusses self-dIALOGue and its role in exploring thoughts, which is more about introspection than focus on senses or aesthetics.

    So, the topic that clearly emphasizes discipline in sensory focus and aesthetic enhancement is "Discipline of Sensory Focus." It's directly addressing how focusing on each sense can lead to a deeper appreciation of beauty and inner peace.
    </think>

    The topic that emphasizes the discipline of sensory focus and aesthetic enhancement is **"Discipline of Sensory Focus"**. This section discusses techniques such as deep breathing, meditation, and other methods aimed at enhancing the experience through focused attention on one's senses and fostering a deeper connection with the world.

    Global Search:
    <think>
    Okay, I need to figure out how the themes in this knowledge base relate to personal growth and moderation. Let me start by looking at each topic one by one.

    First up is "Existential Void," which talks about a hole or lack of satisfaction. That seems like a clear reflection of not having been satisfied enough. Maybe it's about finding meaning or purpose, so that ties into personal growth.

    Next is "Mental Quacks" under Buddhism. The topics here are mental qualities and things like tranquility and cognitive flexibility. These could be seen as parts of maintaining inner peace, which is a form of moderation since they're about balance rather than excessive engagement with tasks.

    Then there's "Meditation Benefits." This includes mindfulness practices and how meditation affects the brain. Meditation itself is a form of personal growth because it helps in self-awareness and reflection. So that ties into both mental and emotional aspects, promoting balance through structured practice.

    "Power Dynamics" under social media topics is about perspectives shifting based on others' intentions. That’s more about community or superficial interactions, which might not directly show personal growth but could reflect the ability to navigate different viewpoints, maybe a form of moderation by considering diverse opinions rather than one.

    Looking at "Chronic Issues," it's about dealing with something that doesn't improve over time, like mental fatigue. This can be viewed as a way to grow by finding solutions or new perspectives, balancing continued struggle against progress.

    "Impathy and Connection" is about understanding others' feelings, which involves empathy and maintaining relationships. These are fundamental social skills, part of personal growth in being able to connect with others and contribute to shared well-being.

    "Time Management" under sources is about prioritizing tasks and tracking time. This helps in personal development by improving efficiency, allowing for more focus on activities that matter more, promoting moderation through structured planning.

    Lastly, "Fascist Influences" deals with how certain beliefs affect decisions. This ties into the idea of making choices based on societal or cultural norms, which can lead to growth but may also result from systemic constraints.

    Overall, each theme seems to offer a way to navigate life challenges or social issues by finding balance and meaningful connections, thus reflecting personal growth and moderation in various contexts.
    </think>

    The themes in this knowledge base reflect broader concepts of personal growth and moderation across different topics. Here's how they can be understood in terms of personal development:

    1. **Existential Void**: This theme highlights the idea of not having been satisfied or fulfilled enough. It serves as a reflection of personal growth, where it is essential to find meaning and purpose through meaningful relationships, self-reflection, and action.

    2. **Mental Quacks and Mindfulness**: Topics like "traineuse" (lack of mental flexibility) and the role of mindfulness in meditating on thoughts suggest that mental quacks can be part of personal growth. These practices help maintain balance by fostering inner clarity and freedom from excessive focus, which is crucial for maintaining well-being.

    3. **Meditation Benefits**: The emphasis on meditation as a form of personal growth indicates its role in self-awareness and reflection. Meditation helps individuals navigate their lives more effectively, allowing them to find contentment and growth while prioritizing inner peace over superficial tasks.

    4. **Power Dynamics and Perspectives Shifts**: This theme reflects the complexity of navigating social interactions where perspectives can change based on others' intentions. It is a form of moderation that allows individuals to explore diverse viewpoints and maintain personal autonomy in relationships or communities.

    5. **Chronic Issues and Progression**: Topics like chronic issues suggest the importance of finding solutions rather than continuing without progress. This approach mirrors personal growth by encouraging seeking new opportunities for improvement and balance, ensuring continued well-being.

    6. **Impathy and Relationships**: Understanding others' feelings ties into the concept of social connection and mutual support. This reflects a form of moderation where individuals connect on a deeper level, contributing to shared well-being and personal development through empathy and active participation in relationships.

    7. **Time Management and Prioritization**: Managing time effectively is crucial for personal growth as it allows individuals to prioritize tasks, focusing on what matters most while still maintaining balance with other responsibilities.

    8. **Fascist Influences**: Acknowledging the impact of societal or cultural norms on choices reflects a broader consideration of how external factors can influence personal decisions and growth. It underscores the importance of critically examining influences rather than being driven by them.

    In summary, these themes collectively illustrate how personal growth involves navigating challenges, finding balance through various perspectives, and maintaining well-being. They reflect the ability to adjust behavior in response to changing circumstances while fostering resilience and adaptability—key elements of personal development and moderation.

    Hybrid Search:
    <think>
    Okay, I'm trying to figure out how all these topics in the knowledge base connect to self-expression, beauty, and resource management. Let me start by reading through each topic one by one.

    The first topic is about existential void and emotional dissatisfaction. It talks about needing inner fulfillment but feeling bad when you feel bad. That makes sense because it's about finding meaning and balance—self-expression. The beauty part might come from the sense of purpose or completeness, which are forms of beauty in this context. Resource management could be about how we manage our time and resources to achieve these states.

    Next up is meditation as a way to down with it. I know meditation is great for relaxing and reducing stress, which relates to self-expression by finding peace. Beauty here could mean the calmness or inner tranquility. Resource management might involve using meditation sessions as a way to manage daily tasks without losing time.

    The third topic mentions Nietzsche's will to power. That's about pushing forward and achieving goals, so it ties into resource management when people prioritize their efforts—saving resources by focusing on what's most important. Self-expression here could be through creativity or action that contributes positively. Beauty might relate to the drive to do something beyond our needs.

    Thinking with a friend from the fourth topic sounds like it's about social interaction and how sharing can reduce resource use (like not using the phone while watching TV). It's more about relationships and balance, which is self-expression. Beauty could be in harmony or mutual understanding. Resource management would involve being mindful of what's needed for these interactions.

    Nestrix, the organism, reminds me of humans interacting with nature. The idea that our lives shape nature and vice versa might relate to resource management when we rely on others for resources. Self-expression could come from gratitude or appreciation for natural elements. Beauty is perhaps in the balance between human and natural aspects.

    The fifth topic about the device's role as a tool suggests using resources wisely. The device itself isn't a resource unless it needs power, but managing how it uses resources can affect outcomes. Self-expression might be through effective use of technology. Beauty could involve efficiency or innovation in its design.

    Meditation and life coaching (topics 6-7) seem to focus on self-improvement without using excessive time. Self-expression here is about personal growth and happiness. Beauty could be the sense of calm or progress achieved. Resource management might mean prioritizing one's goals over superficial tasks, which can free up resources for other areas.

    Exploring and bonding (topics 8-9) likely emphasizes community and shared experiences. Self-expression is about contributing to a group. Beauty here could be harmony or mutual enjoyment. Resources could involve time spent on activities that benefit others.

    The sixth topic about the device's role as a tool again ties into managing resources for success, which relates to resource management. Self-expression through using technology effectively. Beauty might be in creating tools that enhance lives. Resource management would mean being efficient with the resources they use.

    Innovation and collaboration (topics 10-13) emphasize teamwork and problem-solving. Self-expression through active participation. Beauty is perhaps creativity or innovation. Resources could involve planning projects together, which saves time when working on larger issues.

    Meditation in spiritual contexts, as mentioned in topics 14-22, suggests mindfulness for personal growth without using much time. Self-expression is about peace and balance. Beauty could be the sense of connection through this practice. Resource management might mean focusing on one's needs without overcomplicating things.

    The last topic about resourcefulness involves prioritizing tasks and using resources wisely to achieve goals. Self-expression through effective planning. Beauty perhaps in being productive or goal-oriented. Resource management is straightforward, like budgeting money for essential activities.

    Putting it all together, each topic seems to tie into self-expression by focusing on balance, creativity, and purpose. Beauty could refer to the sense of completion, satisfaction, or harmony achieved through these practices. Resource management might be about using time wisely to support these aspects without overextending resources.
    </think>

    The knowledge base provides a comprehensive exploration of how concepts like self-expression, beauty, and resource management intersect across various topics. Here's a structured summary:

    1. **Existential Void and Emotional Dissatisfaction**: This topic emphasizes the need for inner fulfillment and balancing one's life to avoid feelings of emotional dissatisfaction. Self-expression is about finding meaning and purpose, with beauty potentially referring to this sense of completion.

    2. **Meditation as a Tool for Downwith It**: Meditation promotes relaxation and reducing stress, aligning with self-expression by fostering peace. Beauty could be the calmness or inner tranquility gained through practice. Resource management might involve using meditation sessions efficiently without losing time.

    3. **Nietzschean Will to Power**: This concept highlights prioritizing goals and efforts over superficial tasks, supporting resource management by focusing on what's most important. Self-expression is through effective action, with beauty in the drive to achieve purposeful goals.

    4. **Thinking with a Friend**: Social interactions reduce resource use, emphasizing community and shared experiences. Beauty could be harmony or mutual understanding. Resource management involves prioritizing one's needs without overextending resources.

    5. **Nestrix and Human Interaction**: Interactions with nature balance human and natural aspects. Self-expression through gratitude and appreciation. Beauty might involve the balance between humans and nature.

    6. **Device as a Tool**: Managing resources for success, supporting self-expression through effective use of technology. Beauty could be efficiency or innovation in design. Resource management involves prioritizing goals over superficial tasks.

    7. **Exploring and Bonding**: Community support fosters shared experiences, enhancing self-expression and community. Beauty is harmony or mutual enjoyment. Resources could involve time spent on activities that benefit others.

    8. **Innovation and Collaboration**: Focuses on teamwork and problem-solving, supporting self-expressivity through active participation. Beauty involves creativity or innovation. Resource management might involve planning projects together for efficiency.

    9. **Meditation in Spiritual Contexts**: Mindful practice for peace without excessive use, aligning with self-expression through peace. Beauty could be the sense of connection. Resource management is straightforward budgeting for essential activities.

    10. **Resourcefulness and Prioritization**: Priorities in task management support self-expression through effective planning. Beauty might be productivity or goal-oriented focus. Resource management involves efficient use without neglecting resources.

    In summary, each topic explores how balance (self-expression), completeness (beauty), and efficiency (resource management) are interconnected, emphasizing the need for mindfulness, collaboration, and harmony in various contexts.
    <think>
    Okay, I'm trying to figure out how all these topics in the knowledge base connect to self-expression, beauty, and resource management. Let me start by reading through each topic one by one.

    The first topic is about existential void and emotional dissatisfaction. It talks about needing inner fulfillment but feeling bad when you feel bad. That makes sense because it's about finding meaning and balance—self-expression. The beauty part might come from the sense of purpose or completeness, which are forms of beauty in this context. Resource management could be about how we manage our time and resources to achieve these states.

    Next up is meditation as a way to down with it. I know meditation is great for relaxing and reducing stress, which relates to self-expression by finding peace. Beauty here could mean the calmness or inner tranquility. Resource management might involve using meditation sessions as a way to manage daily tasks without losing time.

    The third topic mentions Nietzsche's will to power. That's about pushing forward and achieving goals, so it ties into resource management when people prioritize their efforts—saving resources by focusing on what's most important. Self-expression here could be through creativity or action that contributes positively. Beauty might relate to the drive to do something beyond our needs.

    Thinking with a friend from the fourth topic sounds like it's about social interaction and how sharing can reduce resource use (like not using the phone while watching TV). It's more about relationships and balance, which is self-expression. Beauty could be in harmony or mutual understanding. Resource management would involve being mindful of what's needed for these interactions.

    Nestrix, the organism, reminds me of humans interacting with nature. The idea that our lives shape nature and vice versa might relate to resource management when we rely on others for resources. Self-expression could come from gratitude or appreciation for natural elements. Beauty is perhaps in the balance between human and natural aspects.

    The fifth topic about the device's role as a tool suggests using resources wisely. The device itself isn't a resource unless it needs power, but managing how it uses resources can affect outcomes. Self-expression might be through effective use of technology. Beauty could involve efficiency or innovation in its design.

    Meditation and life coaching (topics 6-7) seem to focus on self-improvement without using excessive time. Self-expression here is about personal growth and happiness. Beauty could be the sense of calm or progress achieved. Resource management might mean prioritizing one's goals over superficial tasks, which can free up resources for other areas.

    Exploring and bonding (topics 8-9) likely emphasizes community and shared experiences. Self-expression is about contributing to a group. Beauty here could be harmony or mutual enjoyment. Resources could involve time spent on activities that benefit others.

    The sixth topic about the device's role as a tool again ties into managing resources for success, which relates to resource management. Self-expression through using technology effectively. Beauty might be in creating tools that enhance lives. Resource management would mean being efficient with the resources they use.

    Innovation and collaboration (topics 10-13) emphasize teamwork and problem-solving. Self-expression through active participation. Beauty is perhaps creativity or innovation. Resources could involve planning projects together, which saves time when working on larger issues.

    Meditation in spiritual contexts, as mentioned in topics 14-22, suggests mindfulness for personal growth without using much time. Self-expression is about peace and balance. Beauty could be the sense of connection through this practice. Resource management might mean focusing on one's needs without overcomplicating things.

    The last topic about resourcefulness involves prioritizing tasks and using resources wisely to achieve goals. Self-expression through effective planning. Beauty perhaps in being productive or goal-oriented. Resource management is straightforward, like budgeting money for essential activities.

    Putting it all together, each topic seems to tie into self-expression by focusing on balance, creativity, and purpose. Beauty could refer to the sense of completion, satisfaction, or harmony achieved through these practices. Resource management might be about using time wisely to support these aspects without overextending resources.
    </think>

    The knowledge base provides a comprehensive exploration of how concepts like self-expression, beauty, and resource management intersect across various topics. Here's a structured summary:

    1. **Existential Void and Emotional Dissatisfaction**: This topic emphasizes the need for inner fulfillment and balancing one's life to avoid feelings of emotional dissatisfaction. Self-expression is about finding meaning and purpose, with beauty potentially referring to this sense of completion.

    2. **Meditation as a Tool for Downwith It**: Meditation promotes relaxation and reducing stress, aligning with self-expression by fostering peace. Beauty could be the calmness or inner tranquility gained through practice. Resource management might involve using meditation sessions efficiently without losing time.

    3. **Nietzschean Will to Power**: This concept highlights prioritizing goals and efforts over superficial tasks, supporting resource management by focusing on what's most important. Self-expression is through effective action, with beauty in the drive to achieve purposeful goals.

    4. **Thinking with a Friend**: Social interactions reduce resource use, emphasizing community and shared experiences. Beauty could be harmony or mutual understanding. Resource management involves prioritizing one's needs without overextending resources.

    5. **Nestrix and Human Interaction**: Interactions with nature balance human and natural aspects. Self-expression through gratitude and appreciation. Beauty might involve the balance between humans and nature.

    6. **Device as a Tool**: Managing resources for success, supporting self-expression through effective use of technology. Beauty could be efficiency or innovation in design. Resource management involves prioritizing goals over superficial tasks.

    7. **Exploring and Bonding**: Community support fosters shared experiences, enhancing self-expression and community. Beauty is harmony or mutual enjoyment. Resources could involve time spent on activities that benefit others.

    8. **Innovation and Collaboration**: Focuses on teamwork and problem-solving, supporting self-expressivity through active participation. Beauty involves creativity or innovation. Resource management might involve planning projects together for efficiency.

    9. **Meditation in Spiritual Contexts**: Mindful practice for peace without excessive use, aligning with self-expression through peace. Beauty could be the sense of connection. Resource management is straightforward budgeting for essential activities.

    10. **Resourcefulness and Prioritization**: Priorities in task management support self-expression through effective planning. Beauty might be productivity or goal-oriented focus. Resource management involves efficient use without neglecting resources.

    In summary, each topic explores how balance (self-expression), completeness (beauty), and efficiency (resource management) are interconnected, emphasizing the need for mindfulness, collaboration, and harmony in various contexts.

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

``` python
import asyncio

# Reinitialize the RAG instance; this will load data from the existing working directory "./dickens"
rag = asyncio.run(initialize_rag())

# Example: perform a query to extract data from the Dickens working directory
query_text = "Please extract and summarize all topics and themes stored in the Dickens dataset."
result = rag.query(query_text, param=QueryParam(mode="global"))

# If the result is a streaming response, consume it asynchronously; otherwise, print the result directly.
if hasattr(result, '__aiter__'):
    async def consume_stream():
        full_response = ""
        async for chunk in result:
            full_response += chunk
        print(full_response)
    asyncio.run(consume_stream())
else:
    print(result)
```

    NameError: name 'initialize_rag' is not defined
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mNameError[0m                                 Traceback (most recent call last)
    Cell [0;32mIn[9], line 4[0m
    [1;32m      1[0m [38;5;28;01mimport[39;00m [38;5;21;01masyncio[39;00m
    [1;32m      3[0m [38;5;66;03m# Reinitialize the RAG instance; this will load data from the existing working directory "./dickens"[39;00m
    [0;32m----> 4[0m rag [38;5;241m=[39m asyncio[38;5;241m.[39mrun([43minitialize_rag[49m())
    [1;32m      6[0m [38;5;66;03m# Example: perform a query to extract data from the Dickens working directory[39;00m
    [1;32m      7[0m query_text [38;5;241m=[39m [38;5;124m"[39m[38;5;124mPlease extract and summarize all topics and themes stored in the Dickens dataset.[39m[38;5;124m"[39m

    [0;31mNameError[0m: name 'initialize_rag' is not defined

``` python
# Assuming the `rag` instance has already been created in a previous cell

print("\nAdditional Questions:")

# Naive mode query: exploring economic aspects
print("Naive Query:")
print(
    rag.query(
        "What topic discusses economic prosperity and its impact on community life?",
        param=QueryParam(mode="naive")
    )
)

# Local mode query: connecting artistic expression with systematic resource allocation
print("\nLocal Query:")
print(
    rag.query(
        "Which topic integrates artistic expression with systematic resource allocation?",
        param=QueryParam(mode="local")
    )
)

# Global mode query: comprehensive analysis of aesthetic themes
print("\nGlobal Query:")
print(
    rag.query(
        "Can you provide a comprehensive analysis of aesthetic themes across all topics?",
        param=QueryParam(mode="global")
    )
)
```


    Additional Questions:
    Naive Query:
    <think>
    Alright, I'm looking at this query where the user is asking about the topic discussing economic prosperity and its impact on community life. They've provided a document chunk with several topics related to existential void, meditation, and various aspects of human psychology and spirituality.

    First, I need to determine which of these topics specifically relates to economics or economic prosperity. The document chunks cover a wide range from existential themes to social and spiritual topics. However, none directly mention anything about economics or how economic factors affect communities.

    Since the provided documents don't include information on economic prosperity, I can conclude that this topic isn't covered in the given chunks. Therefore, there's no relevant discussion here, and the user is looking for more content than is available.
    </think>

    This query does not provide any information about economic prosperity or its impact on community life as it relates to the topics covered in the document chunks provided. The document chunk discusses various themes related to existential void, meditation, spirituality, and human psychology, but none of these topics cover economics or its effects on communities. Therefore, there is no relevant discussion available for this query based on the given information.

    Local Query:
    <think>
    Okay, I'm trying to figure out which topic in the given Knowledge Base integrates artistic expression with systematic resource allocation. Let's break this down step by step.

    First, I'll go through each topic and see what they're about. The Topics include things like existential void, meditation benefits, cognitive health, time management, philosophy of resource allocation, etc. My goal is to find a topic that connects both artistic expression and systematic resource allocation.

    Looking at the provided Knowledge Base, I notice topics related to meditation and its effects on people's lives. Specifically, there are sections about meditation being helpful for various forms and the benefits it can provide. This seems relevant because meditation involves systematic practice aimed at inner peace, which is a form of artistic expression in that you're channeling yourself into a state of calm.

    Another topic mentioned is resource allocation itself, but it's more about how to manage resources systematically. However, I'm looking for something that combines both. The meditation section might be the key here because it emphasizes systematic practice and artistry through the process of meditating.

    I also see sections on existential themes like the void and inner fulfillment, which can tie into artistic expression by creating a sense of peace or connection. But these are more about themes than methods of expression. 

    The Time Management section discusses practical techniques for time allocation, but again, it's about applying resources (time) systematically rather than expressing them artistically.

    The philosophy of resource allocation deals with structuring resources, which is methodical but doesn't directly connect to artistic expression unless we consider how the allocation process itself can be seen as an artistic creation or work in progress. But this seems less direct compared to meditation's focus on inner state and practice.

    Given all these points, the meditation section stands out because it deals with systematic resource allocation (time and energy) while creating an artistic expression of peace and inner contentment. This aligns perfectly with what I'm looking for.
    </think>

    The topic that best integrates artistic expression with systematic resource allocation is the **Meditation Section** within the provided Knowledge Base. This section emphasizes the artful application of time and energy through meditation, aiming to achieve an inner state of peace and calm. Through systematic practice, meditants encourage creative expression in creating a sense of inner freedom and connection, aligning with both artistic creation and the structured allocation of resources.

    Global Query:
    <think>
    Alright, let's see what I need to do here. The  asked me to provide a comprehensive analysis of aesthetic themes across all topics in their provided knowledge base. Hmm, okay, so I have to go through each topic they've given and extract the key aesthetic elements from each. 

    First, I should review the entities and relationships to make sure I haven't missed anything. The entities are things like >Jordan, |Alex, Taylor, and various types like UNKNOWN. The relationships involve how these entities interact with each other, especially focusing on timestamps for when this information was acquired. Each relationship has a description of the theme it explores.

    Looking at the topics provided, they cover a wide range from existential themes to more personal or philosophical ones. I think I should go through each topic one by one and extract aesthetic elements. For each, maybe include some key characteristics like nature, symbolism, technology, etc., based on the content. 

    Wait, but how do I know what makes something an aesthetic theme? Aesthetic themes are usually about beauty, form, perception, or spiritual impact. So I need to identify which parts of the topics fall under these categories. For example, in Jordan's interactions with Taylor and Alex, there might be themes related to power dynamics, mutual respect, etc.

    Also, since the  mentioned not to include anything not provided by the knowledge base, I have to make sure my analysis strictly follows what's given without making any assumptions beyond that.

    I should structure this into sections for each topic, maybe with headings like Nature, Symbolism, Technology, and so on. Each section will list aesthetic elements specific to that category as per the entities and relationships provided.

    Let me start by outlining each topic and extracting their aesthetic themes. For example, in Topic 23 about meditation benefits, I can note how it promotes mindfulness practices, mutual respect, etc., under the nature or symbolism umbrella.

    I also need to remember the Response Rules: multiple paragraphs, markdown headings, no added information not in knowledge base, and maintain continuity with the conversation history. So after structuring each topic's aesthetic themes, I'll wrap it up neatly without any fluff.

    Wait, but some topics might have more than one aesthetic theme. For example, existential void could encompass power dynamics, perspective shifts, etc. I need to ensure each aesthetic element is clearly defined under its respective category.

    Also, since the knowledge base includes timestamps for when this info was created, maybe that's relevant in explaining how these themes developed or co-evolved over time. But the  didn't specify including timestamps in the response rules, so perhaps it's better not to include them unless necessary.

    Finally, I should make sure my analysis is concise and fits within multiple paragraphs as per the response format. Each aesthetic theme should be a clear bullet point under its respective category, making it easy to read and understand.
    </think>

    ### Comprehensive Analysis of Aesthetic Themes Across Topics

    #### Topic 22: Existential Void – i.e., a hole in reality, satisfaction hard to come by, don’t feel bad that you feel bad  
    **Aesthetic Themes:**  
    - **Nature:** Emphasizes the absence of purpose or meaningful connection.  
    - **Symbolism:** Often represents the void that arises when expectations and reality clash.  
    - **Perception:** Highlights the need for acceptance in a perceived hole in existence.  

    #### Topic 23: Meditation Benefits, Mindfulness Practices, Mental Health, Spiritual Techniques, Success Strategies  
    **Aesthetic Themes:**  
    - **Nature:** Focuses on physical health benefits of mindfulness and relaxation.  
    - **Technology:** Explores how technology can enhance mental well-being through specific practices.  
    - **Symbolism:** Suggests the use of symbols to foster clarity and emotional balance.  
    - **Perception:** Encourages a holistic approach to self-improvement, integrating various perspectives.  

    #### Topic 24: Concentration, Awareness, Productivity, Mental Discipline, Success Strategies  
    **Aesthetic Themes:**  
    - **Nature:** Drives towards deep focus and sustained concentration for efficiency.  
    - **Technology:** Explores the role of modern tools in enhancing attention and productivity.  
    - **Symbolism:** May include the importance of patience and precision in achieving goals.  
    - **Perception:** Highlights the value of mental discipline in overcoming challenges.  

    #### Topic 25: Nature Symbolism, Astrological Essence, Physical Characteristics, Spiritual Techniques, Environmental Connection  
    **Aesthetic Themes:**  
    - **Nature:** Depicts the symbolic significance of natural elements and their influence on human experiences.  
    - **Symbolism:** Focuses on the essence of nature as a metaphor for spiritual or existential growth.  
    - **Technology:** May not be relevant here but could relate to data-driven insights in related fields.  
    - **Perception:** Encourages viewing nature through symbolic lenses rather than literal ones.  

    #### Topic 26: Psycho vs. socio Path – One more productive, both are cranks  
    **Aesthetic Themes:**  
    - **Nature:** Emphasizes the importance of balanced social interactions for personal growth.  
    - **Symbolism:** Highlights the tension between psychoanalysis and socialism.  
    - **Technology:** May not be applicable as it pertains to abstract themes rather than practical outcomes.  
    - **Perception:** Encourages critical thinking about the nature of identity and progress.  

    #### Topic 27: Paradox with In – Complexity with In Is Tantamount to Paradoxical Traits and In Behavioral Incongruity is the Norm  
    **Aesthetic Themes:**  
    - **Nature:** Focuses on paradoxical yet balanced aspects of existence.  
    - **Technology:** May relate to technological innovations in addressing complexity or behavior.  
    - **Symbolism:** Encourages a perspective that sees paradox as a sign of harmony rather than chaos.  
    - **Perception:** Highlights the importance of understanding and navigating complex relationships.  

    #### Topic 28: Spiritual Presence, Present Focus, Divine Immanence, Specificity with Human Nature  
    **Aesthetic Themes:**  
    - **Nature:** Reflects on the essence of being in the present moment.  
    - **Symbolism:** Suggests spiritual integration as a key aesthetic.  
    - **Technology:** May not be directly relevant but could relate to data-driven insights in related fields.  
    - **Perception:** Emphasizes the importance of inner focus and awareness in achieving spiritual goals.  

    #### Topic 29: Time – Don’t Be afraid – Measure It with Stopwatches, Clocks, Calimers – Be Intimate With Time  
    **Aesthetic Themes:**  
    - **Nature:** Highlights the concept of continuous time and its significance.  
    - **Symbolism:** Encourages viewing time as a fundamental aspect of existence rather than an abstract tool.  
    - **Technology:** May not be directly relevant but could relate to data-driven insights in related fields.  
    - **Perception:** Emphasizes valuing time for personal and emotional growth.  

    #### Topic 30: Feminism – Gitty Up! Update 2-19-19 Feminism is important to me because I see half the population of the world suffering from a patriarchical prejudice that is stunting the growth of civilization. If that’s not enough to be a feminist, I don’t know what is. Prejudice is inherently dumb. Patriarchy is a biased archaic system. Any ways I say gitty up because its little traveled territory for men to have a legitimate opinion on feminist issues, but if you see something – take the concept of full expression as an example of maximized beauty, we see its extreme importance in all of our lives. Older but mildly edited writing – It’s about comfort to express fully – inequality runs deep – share your beautiful life – grrrl! Be all you and that’s all you need to be satisfied… But for real though sex positive feminism…. can’t say enough about it. You see in sexuality there's a power of agenda (i.e. who sets the agenda is the powerful one). In that sense, feminist sexuality exists wherever women are controlling or have an equal role in the sexual agenda. I would also argue that feminist sexuality exists wherever women are satisfied with the sexual agenda or satisfied with their decision to engage in a sexual act even if they’re totally lacking any control of the agenda. Update 2-19-19 – Is there a backlash against sex positivity in 2019? Could it be just a general distaste in men? Or could it be the influence of fascism in the world? Are we subtly acting more fascist towards how we approach sex?  

    **Aesthetic Themes:**  
    - **Nature:** Focuses on the universal themes of power dynamics and social hierarchies.  
    - **Symbolism:** Highlights the importance of gender balance and equality.  
    - **Technology:** May not be directly relevant but could relate to data-driven insights in related fields.  
    - **Perception:** Encourages valuing the role of women in maintaining societal norms and behaviors.  

    This analysis provides a structured, comprehensive look at aesthetic themes across all topics, focusing on nature, symbolism, technology, and philosophical depth.

``` python
import time

def print_query_metrics(query_text, mode):
    start_time = time.time()
    result = rag.query(query_text, param=QueryParam(mode=mode))
    duration = time.time() - start_time
    # Check if the result is an async generator; if so, run the stream synchronously for metrics.
    if hasattr(result, '__aiter__'):
        async def get_full_response():
            response = ""
            async for chunk in result:
                response += chunk
            return response
        result = asyncio.run(get_full_response())
    print(f"Query: {query_text}")
    print(f"Mode: {mode}")
    print("Response:", result)
    print(f"Time taken: {duration:.2f} seconds\n")

print("\nMetrics for Additional Queries:")

# Example metric measurement for a naive mode query
print_query_metrics(
    "How does the system interpret the integration of self-expression with economic factors?", 
    "naive"
)

# Example metric measurement for a global mode query
print_query_metrics(
    "Can you detail the thematic evolution in topics emphasizing moderation?", 
    "global"
)
```


    Metrics for Additional Queries:
    Query: How does the system interpret the integration of self-expression with economic factors?
    Mode: naive
    Response: <think>
    Okay, let me try to figure out how the system interprets the integration of self-expression with economic factors based on the provided document chunks and conversation history.

    First, I need to understand what "self-expression" means in this context. From the document, it seems like it refers to expressing personal thoughts, values, and ideas through various channels. The user mentioned things like talking about themselves, exercising, etc., which all contribute to self-expression.

    Now, looking at the conversation history, topics are discussed around existential themes, pragmatism, meditation, abstract concepts, etc. These seem more aligned with a deeper exploration of life's meaning and purpose rather than economics. There are no direct references or mentions of economic factors in these topics.

    In the response rules, there's nothing specific about integrating self-expression with economic factors. The focus is on philosophy, ethics, personal development, etc., without touching on economics or finance aspects. Therefore, it seems the system doesn't interpret any connection between self-expression and economics since both are not covered in the provided document chunks.

    Maybe I should check if there's any implicit assumption here. Perhaps the user expects the system to explore how one can balance their personal expression with economic choices? But without explicit information on that angle, it's hard to say. It might be a scenario where the integration isn't expected or considered at all based on the provided materials.

    Also, considering the context, the document chunks don't mention any real-world applications of self-expression in economics. They are more about personal development and existential themes, which aren't directly related. So, I think it's safe to say that the system treats these topics separately without integrating them with economic factors.
    </think>

    The system does not interpret the integration of self-expression with economic factors based on the provided document chunks. The conversation history discusses existential themes, pragmatism, meditation, and abstract concepts, which do not include any explicit mention of economics or financial topics. Therefore, there is no indication that the system would integrate personal expression with economic considerations in this context.
    Time taken: 20.87 seconds

    Query: Can you detail the thematic evolution in topics emphasizing moderation?
    Mode: global
    Response: <think>
    Okay, I need to figure out how the topics related to moderation have evolved over time based on the provided knowledge base. Let me start by reviewing each section.

    First, looking at "Sources," they cover various aspects of moderation like mental flexibility and spiritual techniques. The 25th topic talks about animal/vegetarian astrology, which is more about nature and essential elements than social or personal moderation. So maybe that's an early shift away from moderation focused on human needs.

    Next, the "Relationships" section has several interactions. One of them mentions Taylor interacting with a device directly, which seems more about communication rather than emotional regulation. Another interaction involves Taylor feeling authoritarian certainty, possibly related to power dynamics but not necessarily centered around moderation in personal or social aspects. These interactions are less focused on balancing emotions and behaviors.

    Looking at the "Knowledge Base," the topics under "Sources" include themes like existential void, meditation benefits, concentration for awareness, and spiritual techniques. The first two topics talk about emotional satisfaction and self-compassion, which seem more about mental well-being rather than moderation in a broader sense. The later topics touch on time and temporal awareness but don't delve into balancing human needs.

    So the main evolution seems to be from early focus on human needs and emotional well-being towards personal growth and spiritual practices. There's a gradual shift away from social or ethical considerations in moderation but increases in the emphasis on psychological aspects of self-awareness and relationships with oneself through various interactions.

    I think I need to structure this response by discussing each topic, highlighting how the content changed from early sources, focusing on emotions and needs to later, incorporating personal growth and spiritual practices. I should mention that while moderation is still a significant theme, it's integrated more into personal development rather than just balancing social or ethical aspects.
    </think>

    The evolution of topics related to moderation in the provided knowledge base reflects a shift from an early emphasis on human needs and emotional well-being to a focus on personal growth, psychological regulation, and spiritual practices. Here is a detailed overview:

    1. **Early Focus (Sources 25-30):**
       - Topics under "Sources" such as animal/vegetarian astrology and nature symbolism emphasize essential elements and natural processes. These discussions were more about understanding the world around them rather than balancing human needs or personal growth.
       - Interactions with Taylor, Taylor's authoritarian certainty, and Alex's emotional dissociation from Taylor reflect early concerns about maintaining social harmony and emotional relationships.

    2. **Shift to Personal and Emotional Well-being:**
       - Topics like existential void, self-compassion, and self-awareness focus more on personal mental flexibility and emotional satisfaction. These discussions began to integrate human needs into the narrative of personal growth.
       - The mention of Taylor feeling authoritarian certainty suggests an interest in understanding relationships and balancing power dynamics, though not directly related to moderation.

    3. **Expansion into Personal Development:**
       - As interactions with Jordan and Alex became more prominent, topics began to incorporate psychological regulation and self-awareness more deeply. For example, Alex's interaction with a device involved communication about mental flexibility.
       - The introduction of concepts like mission evolution, active participation, mutual respect, and perspective shifts indicates an expansion into areas of personal development rather than just emotional balance.

    4. **Incorporation of Personal Growth:**
       - Topics under "Knowledge Base" such as mission evolution emphasize personal growth through interaction with others and internalized knowledge.
       - The phrase "mission evolution" suggests a focus on personal trajectory or goals, which aligns with themes of self-improvement.

    5. **Focus on Spiritual and Psychological Practices:**
       - As interactions with the device involved co-partnership in story development (topics 2), there was an interest in narrative writing, storytelling techniques, and creative expression.
       - Topics like meditation benefits, mindfulness practices, and spiritual techniques began to incorporate more psychological and emotional aspects, reflecting a shift from abstract stories into personal narratives of self-improvement.

    6. **Continual Evolution towards Personal Development:**
       - The recurring theme of "personal development" in interactions with Jordan and Alex highlights the evolution toward a narrative focused on individual growth and progress.
       - Topics like power dynamics and perspective shifts suggest an interest in understanding oneself as an entity, balancing internal and external forces.

    In summary, while moderation concepts were present in earlier sources, they gradually expanded into personal and emotional regulation. The focus shifted from social harmony to personal development through narratives of narrative writing, narrative evolution, and personal growth. This evolution reflects a deeper integration of psychological, spiritual, and emotional aspects into the narrative of personal development.
    Time taken: 33.92 seconds

``` python
# Cell 2: Corrected System Metrics Analysis
import asyncio
import os

async def show_system_metrics():
    # Initialize RAG to access storage components
    rag = await initialize_rag()
    
    # Get document and chunk statistics through proper interface
    try:
        # Check if storage exists in working directory
        doc_path = os.path.join(WORKING_DIR, "documents")
        chunk_path = os.path.join(WORKING_DIR, "chunks")
        
        print("Document Storage Metrics:")
        if os.path.exists(doc_path):
            doc_files = [f for f in os.listdir(doc_path) if f.endswith(".json")]
            print(f"Total Documents: {len(doc_files)}")
        
        print("\nChunk Storage Metrics:")
        if os.path.exists(chunk_path):
            chunk_files = [f for f in os.listdir(chunk_path) if f.endswith(".json")]
            print(f"Total Chunks: {len(chunk_files)}")
            
            # Calculate average chunk size
            total_size = 0
            for f in chunk_files:
                with open(os.path.join(chunk_path, f), "r") as chunk_file:
                    content = chunk_file.read()
                    total_size += len(content)
            if chunk_files:
                avg_size = total_size / len(chunk_files)
                print(f"Average Chunk Size: {avg_size:.2f} characters")

        # Knowledge Graph Metrics
        print("\nKnowledge Graph Metrics:")
        kg_path = os.path.join(WORKING_DIR, "knowledge_graph")
        if os.path.exists(kg_path):
            kg_files = os.listdir(kg_path)
            print(f"Total Graph Files: {len(kg_files)}")
            
    except Exception as e:
        print(f"Error accessing metrics: {str(e)}")

    # General storage analysis
    print("\nStorage Directory Structure:")
    for root, dirs, files in os.walk(WORKING_DIR):
        print(f"{root}:")
        print(f"  Directories: {len(dirs)}")
        print(f"  Files: {len(files)}")

asyncio.run(show_system_metrics())
```

    RuntimeError: asyncio.run() cannot be called from a running event loop
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mRuntimeError[0m                              Traceback (most recent call last)
    Cell [0;32mIn[7], line 52[0m
    [1;32m     49[0m         [38;5;28mprint[39m([38;5;124mf[39m[38;5;124m"[39m[38;5;124m  Directories: [39m[38;5;132;01m{[39;00m[38;5;28mlen[39m(dirs)[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [1;32m     50[0m         [38;5;28mprint[39m([38;5;124mf[39m[38;5;124m"[39m[38;5;124m  Files: [39m[38;5;132;01m{[39;00m[38;5;28mlen[39m(files)[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [0;32m---> 52[0m [43masyncio[49m[38;5;241;43m.[39;49m[43mrun[49m[43m([49m[43mshow_system_metrics[49m[43m([49m[43m)[49m[43m)[49m

    File [0;32m/opt/homebrew/Caskroom/miniforge/base/envs/nunu24/lib/python3.12/asyncio/runners.py:190[0m, in [0;36mrun[0;34m(main, debug, loop_factory)[0m
    [1;32m    161[0m [38;5;250m[39m[38;5;124;03m"""Execute the coroutine and return the result.[39;00m
    [1;32m    162[0m 
    [1;32m    163[0m [38;5;124;03mThis function runs the passed coroutine, taking care of[39;00m
    [0;32m   (...)[0m
    [1;32m    186[0m [38;5;124;03m    asyncio.run(main())[39;00m
    [1;32m    187[0m [38;5;124;03m"""[39;00m
    [1;32m    188[0m [38;5;28;01mif[39;00m events[38;5;241m.[39m_get_running_loop() [38;5;129;01mis[39;00m [38;5;129;01mnot[39;00m [38;5;28;01mNone[39;00m:
    [1;32m    189[0m     [38;5;66;03m# fail fast with short traceback[39;00m
    [0;32m--> 190[0m     [38;5;28;01mraise[39;00m [38;5;167;01mRuntimeError[39;00m(
    [1;32m    191[0m         [38;5;124m"[39m[38;5;124masyncio.run() cannot be called from a running event loop[39m[38;5;124m"[39m)
    [1;32m    193[0m [38;5;28;01mwith[39;00m Runner(debug[38;5;241m=[39mdebug, loop_factory[38;5;241m=[39mloop_factory) [38;5;28;01mas[39;00m runner:
    [1;32m    194[0m     [38;5;28;01mreturn[39;00m runner[38;5;241m.[39mrun(main)

    [0;31mRuntimeError[0m: asyncio.run() cannot be called from a running event loop

``` python
# Cell 2: Final Working Metrics Analysis
import asyncio
import os
import json

async def show_system_metrics():
    rag = await initialize_rag()
    
    print("Analyzing RAG Storage Metrics:\n")
    
    try:
        # Document status analysis
        doc_status_path = os.path.join(WORKING_DIR, "kv_store_doc_status.json")
        if os.path.exists(doc_status_path):
            with open(doc_status_path, "r") as f:
                doc_status = json.load(f)
                print(f"📄 Document Status:")
                print(f"- Total documents tracked: {len(doc_status)}")
                processed = sum(1 for v in doc_status.values() if v.get('processed'))
                print(f"- Processed documents: {processed}")
                print(f"- Processing failures: {sum(1 for v in doc_status.values() if v.get('failed'))}")

        # Chunk storage analysis
        chunk_path = os.path.join(WORKING_DIR, "vdb_chunks.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, "r") as f:
                chunk_data = json.load(f)
                print(f"\n📄 Chunk Storage (vdb_chunks.json):")
                print(f"- Total chunk entries: {len(chunk_data.get('data', []))}")
                if chunk_data.get('data'):
                    avg_size = sum(len(entry.get('text', '')) for entry in chunk_data['data']) / len(chunk_data['data'])
                    print(f"- Average chunk size: {avg_size:.1f} characters")

        # Knowledge graph analysis
        entity_path = os.path.join(WORKING_DIR, "vdb_entities.json")
        rel_path = os.path.join(WORKING_DIR, "vdb_relationships.json")
        
        if os.path.exists(entity_path):
            with open(entity_path, "r") as f:
                entity_data = json.load(f)
                print(f"\n📄 Knowledge Graph Entities (vdb_entities.json):")
                print(f"- Entity records: {len(entity_data.get('data', []))}")
        
        if os.path.exists(rel_path):
            with open(rel_path, "r") as f:
                rel_data = json.load(f)
                print(f"\n📄 Knowledge Graph Relationships (vdb_relationships.json):")
                print(f"- Relationship records: {len(rel_data.get('data', []))}")

    except Exception as e:
        print(f"\n⚠️ Error analyzing metrics: {str(e)}")

    # File system analysis
    print("\n📂 Directory Contents:")
    for item in os.listdir(WORKING_DIR):
        full_path = os.path.join(WORKING_DIR, item)
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            print(f"- {item} ({size} bytes)")

asyncio.run(show_system_metrics())
```

    RuntimeError: asyncio.run() cannot be called from a running event loop
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mRuntimeError[0m                              Traceback (most recent call last)
    Cell [0;32mIn[6], line 61[0m
    [1;32m     58[0m             size [38;5;241m=[39m os[38;5;241m.[39mpath[38;5;241m.[39mgetsize(full_path)
    [1;32m     59[0m             [38;5;28mprint[39m([38;5;124mf[39m[38;5;124m"[39m[38;5;124m- [39m[38;5;132;01m{[39;00mitem[38;5;132;01m}[39;00m[38;5;124m ([39m[38;5;132;01m{[39;00msize[38;5;132;01m}[39;00m[38;5;124m bytes)[39m[38;5;124m"[39m)
    [0;32m---> 61[0m [43masyncio[49m[38;5;241;43m.[39;49m[43mrun[49m[43m([49m[43mshow_system_metrics[49m[43m([49m[43m)[49m[43m)[49m

    File [0;32m/opt/homebrew/Caskroom/miniforge/base/envs/nunu24/lib/python3.12/asyncio/runners.py:190[0m, in [0;36mrun[0;34m(main, debug, loop_factory)[0m
    [1;32m    161[0m [38;5;250m[39m[38;5;124;03m"""Execute the coroutine and return the result.[39;00m
    [1;32m    162[0m 
    [1;32m    163[0m [38;5;124;03mThis function runs the passed coroutine, taking care of[39;00m
    [0;32m   (...)[0m
    [1;32m    186[0m [38;5;124;03m    asyncio.run(main())[39;00m
    [1;32m    187[0m [38;5;124;03m"""[39;00m
    [1;32m    188[0m [38;5;28;01mif[39;00m events[38;5;241m.[39m_get_running_loop() [38;5;129;01mis[39;00m [38;5;129;01mnot[39;00m [38;5;28;01mNone[39;00m:
    [1;32m    189[0m     [38;5;66;03m# fail fast with short traceback[39;00m
    [0;32m--> 190[0m     [38;5;28;01mraise[39;00m [38;5;167;01mRuntimeError[39;00m(
    [1;32m    191[0m         [38;5;124m"[39m[38;5;124masyncio.run() cannot be called from a running event loop[39m[38;5;124m"[39m)
    [1;32m    193[0m [38;5;28;01mwith[39;00m Runner(debug[38;5;241m=[39mdebug, loop_factory[38;5;241m=[39mloop_factory) [38;5;28;01mas[39;00m runner:
    [1;32m    194[0m     [38;5;28;01mreturn[39;00m runner[38;5;241m.[39mrun(main)

    [0;31mRuntimeError[0m: asyncio.run() cannot be called from a running event loop

``` python
def verify_data_ingestion():
    # Check document count matches CSV entries
    with open("./42a.csv") as f:
        csv_rows = sum(1 for _ in csv.DictReader(f))  # Should be 42
        
    with open(os.path.join(WORKING_DIR, "kv_store_doc_status.json")) as f:
        doc_status = json.load(f)
    
    print(f"CSV Entries: {csv_rows}")
    print(f"Ingested Documents: {len(doc_status)}")
    print(f"Processing Success Rate: {sum(1 for v in doc_status.values() if v['processed'])/len(doc_status):.1%}")
```

``` python
def analyze_content_quality():
    with open(os.path.join(WORKING_DIR, "vdb_chunks.json")) as f:
        chunks = json.load(f)['data']
    
    # Verify theme preservation
    theme_coverage = Counter()
    for chunk in chunks:
        if 'themes' in chunk['metadata']:
            for theme in chunk['metadata']['themes']:
                theme_coverage[theme] += 1
                
    print("\nTheme Coverage in Chunks:")
    for theme, count in theme_coverage.most_common(10):
        print(f"- {theme}: {count} mentions")
```

``` python
def validate_knowledge_graph():
    with open(os.path.join(WORKING_DIR, "vdb_entities.json")) as f:
        entities = json.load(f)['data']
    
    with open(os.path.join(WORKING_DIR, "vdb_relationships.json")) as f:
        relationships = json.load(f)['data']
    
    print(f"\nEntity Types: {len({e['metadata']['type'] for e in entities})}")
    print(f"Avg Connections per Entity: {len(relationships)/len(entities):.1f}")
```

``` python
def pipeline_completeness():
    required_files = [
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "kv_store_doc_status.json"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    print("\nMissing Files:" if missing else "\nAll Pipeline Files Present")
    for f in missing:
        print(f"- {f}")
```

``` python
async def test_query_effectiveness(rag):
    test_cases = [
        ("What Nietzsche concepts are criticized?", "global"),
        ("Explain meditation techniques mentioned", "local"),
        ("How is beauty maximized?", "hybrid")
    ]
    
    for question, mode in test_cases:
        print(f"\nTesting '{question}' ({mode}):")
        result = rag.query(question, param=QueryParam(mode=mode))
        print(f"- Top 3 Relevant Chunks:")
        for i, chunk in enumerate(result['chunks'][:3], 1):
            print(f"  {i}. {chunk['metadata']['topic'][:50]}...")
```
