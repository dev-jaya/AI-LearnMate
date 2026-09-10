"""Provider abstraction and validated local question generation for AI LearnMate."""
import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import settings

SUBJECTS = [
    "C", "C++", "Java", "Python", "Data Structures", "Algorithms", "DBMS",
    "Operating Systems", "Computer Networks", "Computer Organization", "Software Engineering",
    "Web Development", "Artificial Intelligence", "Machine Learning", "Cybersecurity", "Cloud Computing",
]

QUESTION_TYPES = {
    "conceptual", "code-output", "debugging", "scenario", "comparison",
    "reasoning", "terminology", "practical", "application", "problem-solving",
}

CONCEPTS = {
    "Python": [
        ("Which keyword defines a function in Python?", ["func", "def", "function", "define"], 1, "Python uses def to define functions.", "Functions"),
        ("Which Python collection stores key-value pairs?", ["list", "tuple", "dictionary", "set"], 2, "Dictionaries map keys to values.", "Collections"),
        ("What does len([10, 20, 30]) return?", ["2", "3", "4", "30"], 1, "The list contains three elements.", "Built-ins"),
        ("Which symbol starts a single-line Python comment?", ["//", "#", "/*", "--"], 1, "Python uses # for comments.", "Syntax"),
        ("Which Python value is immutable?", ["list", "dictionary", "set", "tuple"], 3, "Tuples cannot be changed after creation.", "Data types"),
        ("What does a Python list comprehension create?", ["A compact list", "A database", "A class", "A socket"], 0, "List comprehensions construct lists from iterable expressions.", "Comprehensions"),
        ("Which block handles a Python exception?", ["catch", "except", "rescue", "handle"], 1, "Python uses except with try for exception handling.", "Exceptions"),
        ("What does self refer to in an instance method?", ["The current object", "The parent module", "The return value", "The interpreter"], 0, "self refers to the current instance.", "OOP"),
        ("What is the output of print(2 ** 3)?", ["5", "6", "8", "9"], 2, "The exponentiation operator raises 2 to the third power.", "Operators", "code-output"),
        ("Which statement correctly opens a file for reading?", ["open('data.txt', 'r')", "read('data.txt')", "file('data.txt', 'read')", "open.read('data.txt')"], 0, "open accepts the path and the r mode for reading.", "Files", "practical"),
        ("What error occurs when code uses an undefined variable?", ["NameError", "TypeError", "IndexError", "KeyError"], 0, "Python raises NameError when a name is not defined.", "Exceptions", "debugging"),
        ("Which expression creates a shallow copy of a list named values?", ["values.copy()", "copy(values, deep)", "values.clone(deep)", "list.copyOf(values, 2)"], 0, "list.copy returns a shallow copy.", "Collections", "application"),
        ("Which Python tool installs packages from the package index?", ["pip", "pathlib", "pytest", "pydoc"], 0, "pip installs and manages Python packages.", "Tooling", "terminology"),
        ("What happens when a list index is outside its bounds?", ["IndexError", "The list grows", "None is returned", "KeyError"], 0, "Python raises IndexError for an invalid list index.", "Collections", "reasoning"),
        ("Which construct ensures cleanup code runs after a try block?", ["finally", "else", "cleanup", "ensure"], 0, "finally runs whether an exception occurs or not.", "Exceptions", "conceptual"),
        ("Which decorator commonly defines a static method?", ["@staticmethod", "@static", "@method(static)", "@classless"], 0, "staticmethod creates a method that does not receive an instance automatically.", "OOP", "terminology"),
        ("Which Python type is best for a fixed ordered record?", ["tuple", "set", "dictionary", "generator"], 0, "Tuples represent fixed ordered collections.", "Data types", "application"),
        ("Which expression filters even numbers from values?", ["[x for x in values if x % 2 == 0]", "filter even values", "values.where(even)", "[even(values)]"], 0, "The list comprehension keeps values whose remainder is zero.", "Comprehensions", "code-output"),
        ("What is the main benefit of a generator?", ["Lazy iteration", "Automatic sorting", "Database storage", "Static typing"], 0, "Generators produce values lazily and can save memory.", "Iterators", "comparison"),
        ("Which module provides regular-expression operations?", ["re", "regexlib", "pattern", "textmatch"], 0, "Python's re module provides regular-expression operations.", "Modules", "practical"),
        ("What is the value of bool([])?", ["True", "False", "None", "0.5"], 1, "An empty list is falsy, so bool([]) returns False.", "Truthiness", "code-output"),
        ("Which method adds one item to the end of a list?", ["append", "extend_one", "push_back_only", "insert_end_only"], 0, "append adds one item to the end of a list.", "Lists", "terminology"),
        ("What is the purpose of __init__ in a Python class?", ["Initialize an instance", "Delete a module", "Start a thread only", "Compile a function"], 0, "__init__ initializes a newly created instance.", "OOP", "conceptual"),
        ("Which keyword returns a value from a function?", ["yield", "return", "send", "give"], 1, "return sends a result back to the caller.", "Functions", "terminology"),
        ("What does range(3) produce for iteration?", ["0, 1, 2", "1, 2, 3", "0, 1, 2, 3", "Only 3"], 0, "The stop value is exclusive, so range(3) yields zero through two.", "Loops", "code-output"),
        ("Which keyword creates a custom exception class relationship?", ["class", "exception", "raiseclass", "error"], 0, "Custom exceptions are defined with class, usually inheriting from Exception.", "Exceptions", "practical"),
        ("What does dictionary.get('missing') return by default?", ["None", "KeyError", "False always", "The key name"], 0, "get returns None when the key is absent unless a default is supplied.", "Dictionaries", "reasoning"),
        ("Which operation combines two sets without duplicates?", ["Union", "Intersection only", "Indexing", "Zipping"], 0, "Set union combines members and removes duplicates.", "Sets", "comparison"),
        ("What is recursion?", ["A function calling itself", "A loop without a condition", "A module import", "A list copy"], 0, "Recursion solves a problem through smaller calls to the same function.", "Recursion", "conceptual"),
        ("Which practice prevents a recursive function from calling forever?", ["A base case", "A global variable", "A print statement", "A second import"], 0, "A base case stops recursive expansion.", "Recursion", "debugging"),
        ("What does import math make available?", ["Names from the math module", "All operating-system processes", "A database connection", "A web server"], 0, "import binds a module so its names can be used.", "Modules", "application"),
        ("Which keyword gives a function a fallback parameter value?", ["default", "An assignment in the signature", "fallback", "optional"], 1, "A parameter such as limit=10 defines a default value.", "Functions", "practical"),
        ("What does enumerate(items) provide?", ["Index and value pairs", "Only values", "Sorted values", "Keys from a database"], 0, "enumerate produces an index and value on each iteration.", "Loops", "comparison"),
        ("Which object is used to manage a with block?", ["A context manager", "A decorator only", "A list iterator only", "A socket buffer"], 0, "A context manager defines setup and cleanup around a with block.", "Resource management", "terminology"),
        ("What is duck typing based on?", ["Supported behavior", "Declared class names only", "Memory address", "Package version"], 0, "Duck typing focuses on what an object can do.", "OOP", "reasoning"),
        ("Which collection preserves insertion order and maps keys to values?", ["dict", "set", "tuple", "range"], 0, "Modern Python dictionaries preserve insertion order.", "Dictionaries", "comparison"),
        ("What does `is` compare in Python?", ["Object identity", "Numeric magnitude only", "String length", "Sorted order"], 0, "is tests whether two references point to the same object.", "Operators", "conceptual"),
        ("Which built-in combines items from iterables position by position?", ["zip", "join", "pair", "merge_list"], 0, "zip creates tuples from corresponding iterable items.", "Iterators", "practical"),
        ("What is a module-level variable commonly called?", ["A global", "A closure only", "A field always", "A parameter"], 0, "A name defined at module scope is global to that module.", "Scope", "terminology"),
        ("Which approach makes a function easier to test?", ["Keep it focused with explicit inputs and outputs", "Hide all state globally", "Mix file I/O into every branch", "Use random results"], 0, "Focused functions with explicit inputs and outputs are easier to test.", "Design", "application"),
    ],
    "Java": [
        ("Which keyword creates an object in Java?", ["make", "new", "create", "object"], 1, "The new operator creates an object.", "Objects"),
        ("Which method is the usual Java entry point?", ["start()", "run()", "main()", "init()"], 2, "Execution begins in main.", "Execution"),
        ("What allows one method name to use different parameters?", ["Overloading", "Overriding", "Casting", "Packaging"], 0, "Overloading changes the parameter list.", "Polymorphism"),
        ("Which access modifier is widest for a member?", ["private", "protected", "public", "default"], 2, "public allows access from other classes.", "Encapsulation"),
        ("Which is not a Java primitive type?", ["int", "double", "String", "char"], 2, "String is a reference type.", "Types"),
        ("Which keyword prevents a class from being inherited?", ["static", "final", "sealed", "private"], 1, "A final class cannot be extended.", "Inheritance"),
        ("Which interface represents a collection of unique elements?", ["List", "Queue", "Set", "Map"], 2, "Set does not allow duplicate elements.", "Collections"),
        ("What does garbage collection reclaim?", ["Unused objects", "Source files", "Threads", "Packages"], 0, "It reclaims unreachable objects.", "Memory"),
    ],
    "C": [
        ("Which symbol terminates a C statement?", [".", ";", ":", "#"], 1, "C statements normally end with a semicolon.", "Syntax"),
        ("Which function prints formatted output in C?", ["echo", "print", "printf", "writeLine"], 2, "printf writes formatted output.", "I/O"),
        ("What does a pointer store?", ["A memory address", "Only a character", "A loop", "A package"], 0, "A pointer stores an address.", "Pointers"),
        ("Which header declares malloc?", ["stdio.h", "stdlib.h", "string.h", "math.h"], 1, "malloc is declared in stdlib.h.", "Memory"),
        ("Which loop always executes its body at least once?", ["for", "while", "do-while", "foreach"], 2, "do-while checks its condition after the body.", "Control flow"),
    ],
    "C++": [
        ("Which feature lets a derived class reuse a base class?", ["Inheritance", "Compilation", "Linking", "Tokenization"], 0, "Inheritance reuses and extends a base class.", "OOP"),
        ("Which operator allocates dynamic memory in C++?", ["alloc", "new", "malloc_only", "create"], 1, "new allocates an object or array.", "Memory"),
        ("Which container stores ordered unique values?", ["vector", "set", "map", "stack"], 1, "set stores unique sorted values.", "STL"),
        ("What does RAII tie resource cleanup to?", ["Object lifetime", "Network speed", "Compiler flags", "File names"], 0, "RAII releases resources as objects leave scope.", "Resource management"),
        ("Which function can be overridden in a derived class?", ["virtual function", "macro", "friend declaration", "namespace"], 0, "Virtual functions support runtime dispatch.", "Polymorphism"),
    ],
    "Data Structures": [
        ("Which structure follows last-in, first-out order?", ["Queue", "Stack", "Graph", "Heap"], 1, "A stack is LIFO.", "Stacks"),
        ("Which structure follows first-in, first-out order?", ["Queue", "Tree", "Stack", "Hash"], 0, "A queue is FIFO.", "Queues"),
        ("What is the average lookup time of a good hash table?", ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 0, "Good hashing gives expected constant-time lookup.", "Hashing"),
        ("Which traversal visits a tree root between its subtrees?", ["Preorder", "Inorder", "Postorder", "Level order"], 1, "Inorder is left, root, right.", "Trees"),
        ("Which structure represents relationships as vertices and edges?", ["Graph", "Array", "Stack", "Tuple"], 0, "Graphs contain vertices connected by edges.", "Graphs"),
    ],
    "Algorithms": [
        ("Which algorithm repeatedly selects the smallest remaining item?", ["Merge sort", "Selection sort", "BFS", "Hashing"], 1, "Selection sort selects the next minimum.", "Sorting"),
        ("What is binary search's time complexity on sorted data?", ["O(1)", "O(log n)", "O(n)", "O(n^2)"], 1, "Binary search halves the search interval.", "Searching"),
        ("Which strategy solves overlapping subproblems with stored results?", ["Greedy", "Dynamic programming", "Randomized", "Brute force"], 1, "Dynamic programming stores subproblem results.", "Optimization"),
        ("Which traversal explores a graph layer by layer?", ["DFS", "BFS", "Dijkstra", "Kruskal"], 1, "BFS uses a queue to explore layers.", "Graphs"),
        ("What does Big-O describe?", ["Asymptotic growth", "Variable names", "CPU brand", "Syntax"], 0, "Big-O describes growth as input size increases.", "Complexity"),
    ],
    "DBMS": [
        ("Which SQL command retrieves rows?", ["SELECT", "PUSH", "READROW", "FETCHFILE"], 0, "SELECT queries rows.", "SQL"),
        ("What does a primary key provide?", ["Row uniqueness", "Encryption", "Compression", "Sorting only"], 0, "A primary key uniquely identifies a row.", "Keys"),
        ("Which normal form removes partial dependency?", ["1NF", "2NF", "3NF", "BCNF"], 1, "Second normal form removes partial dependency.", "Normalization"),
        ("Which property means a transaction is all-or-nothing?", ["Atomicity", "Isolation", "Durability", "Consistency"], 0, "Atomicity prevents partial transaction effects.", "Transactions"),
        ("Which join returns matching rows from both tables?", ["INNER JOIN", "CROSS JOIN", "FULL JOIN", "SELF JOIN"], 0, "INNER JOIN returns matching rows.", "Joins"),
    ],
    "Operating Systems": [
        ("What does a process represent?", ["A program in execution", "A source comment", "A database row", "A cable"], 0, "A process is an executing program.", "Processes"),
        ("Which scheduling algorithm uses a time quantum?", ["Round robin", "FCFS", "SJF", "Priority only"], 0, "Round robin rotates processes by time quantum.", "Scheduling"),
        ("What is virtual memory backed by?", ["Secondary storage", "Only registers", "A compiler", "A keyboard"], 0, "Virtual memory can use disk as an extension of RAM.", "Memory"),
        ("Which condition is required for deadlock?", ["Circular wait", "Compilation", "Caching", "Paging only"], 0, "Circular wait is one of the necessary deadlock conditions.", "Deadlocks"),
        ("What does a system call provide?", ["A program-to-kernel interface", "A UI theme", "A network cable", "A sorting step"], 0, "System calls request kernel services.", "Kernel"),
    ],
    "Computer Networks": [
        ("Which protocol maps domain names to IP addresses?", ["HTTP", "DNS", "FTP", "SMTP"], 1, "DNS resolves domain names.", "Application layer"),
        ("How many layers are in the OSI model?", ["5", "6", "7", "8"], 2, "The OSI model has seven layers.", "Models"),
        ("Which device forwards packets between networks?", ["Hub", "Switch", "Router", "Repeater"], 2, "Routers connect and forward between networks.", "Routing"),
        ("Which protocol is connection-oriented?", ["UDP", "IP", "TCP", "ARP"], 2, "TCP establishes a reliable connection.", "Transport"),
        ("What does an IP address identify?", ["A network interface location", "A password", "A file type", "A CPU instruction"], 0, "An IP address identifies a network interface location.", "Addressing"),
    ],
    "AI/ML": [
        ("Which task predicts a continuous value?", ["Classification", "Regression", "Clustering", "Tokenization"], 1, "Regression predicts numeric values.", "Learning types"),
        ("Which method groups unlabeled examples?", ["Regression", "Clustering", "Classification", "Parsing"], 1, "Clustering discovers groups without labels.", "Unsupervised learning"),
        ("What does a validation set help estimate?", ["Generalization", "Keyboard speed", "File size", "Syntax"], 0, "Validation estimates performance on unseen data during development.", "Evaluation"),
        ("What is an embedding?", ["A numeric representation", "A database lock", "A compiler error", "A network cable"], 0, "Embeddings represent items as numeric vectors.", "Representations"),
        ("What does overfitting mean?", ["Memorizing training patterns", "Using no data", "Sorting too fast", "Deleting labels"], 0, "An overfit model performs well on training data but poorly on new data.", "Generalization"),
    ],
    "Web Development": [
        ("Which language structures a web document?", ["HTML", "SQL", "Bash", "CSV"], 0, "HTML provides document structure.", "Frontend"),
        ("Which CSS property changes text color?", ["font-color", "color", "text-paint", "foreground"], 1, "color sets text color.", "CSS"),
        ("Which HTTP status means not found?", ["200", "301", "404", "500"], 2, "404 means the resource was not found.", "HTTP"),
        ("What does REST commonly expose?", ["Resources through HTTP", "Only desktop windows", "CPU registers", "Binary trees"], 0, "REST models resources and uses HTTP operations.", "Backend"),
        ("Which browser API stores structured client data?", ["IndexedDB", "SMTP", "SSH", "DNS"], 0, "IndexedDB stores structured data in the browser.", "Browser APIs"),
    ],
    "Cybersecurity": [
        ("What does phishing primarily try to steal?", ["Credentials", "Screen pixels", "CPU cycles only", "CSS rules"], 0, "Phishing deceives users into revealing sensitive credentials.", "Threats"),
        ("Which principle grants only needed access?", ["Least privilege", "Open access", "Fail open", "Shared root"], 0, "Least privilege limits permissions to what is necessary.", "Access control"),
        ("What does hashing provide when used correctly?", ["One-way digest", "Reversible encryption", "A network route", "A UI layout"], 0, "A cryptographic hash creates a one-way digest.", "Cryptography"),
        ("Which attack injects code into a database query?", ["SQL injection", "ARP spoofing", "DDoS", "Shoulder surfing"], 0, "SQL injection manipulates unsafely constructed queries.", "Application security"),
        ("What does MFA add to login?", ["Another verification factor", "A faster CPU", "A database index", "A CSS rule"], 0, "MFA requires more than one independent verification factor.", "Identity"),
    ],
    "Computer Organization": [
        ("Which component performs arithmetic and logic operations?", ["ALU", "RAM", "Cache", "Bus"], 0, "The arithmetic logic unit performs arithmetic and logical operations.", "CPU", "conceptual"),
        ("What does cache memory reduce?", ["Average memory access time", "Instruction count always", "Disk capacity", "Network latency"], 0, "Cache keeps frequently used data closer to the CPU.", "Memory", "application"),
        ("Which number system uses base 2?", ["Binary", "Decimal", "Octal", "Hexadecimal"], 0, "Binary uses the digits zero and one.", "Representation", "terminology"),
        ("What does an instruction register hold?", ["The current instruction", "The next process", "A database row", "A network packet only"], 0, "The instruction register stores the instruction being executed.", "CPU", "conceptual"),
        ("Which bus carries the location of a memory item?", ["Address bus", "Data bus", "Control bus", "Expansion bus"], 0, "The address bus carries memory addresses.", "Buses", "comparison"),
    ],
    "Software Engineering": [
        ("What is the main goal of unit testing?", ["Test a small component", "Deploy to production", "Design a network", "Encrypt a disk"], 0, "Unit tests check a small isolated unit of behavior.", "Testing", "conceptual"),
        ("Which model emphasizes short iterative releases?", ["Agile", "Waterfall only", "Big bang", "Spiral-free"], 0, "Agile delivers in iterative increments.", "Processes", "comparison"),
        ("What does version control preserve?", ["A history of changes", "Only passwords", "CPU state", "Network routes"], 0, "Version control records and coordinates changes.", "Tools", "application"),
        ("Which quality attribute describes ease of modification?", ["Maintainability", "Availability", "Throughput", "Latency"], 0, "Maintainability concerns future modification effort.", "Quality", "terminology"),
        ("What does code review primarily improve?", ["Defect detection and shared understanding", "Internet speed", "Disk capacity", "Screen resolution"], 0, "Review catches defects and spreads knowledge.", "Collaboration", "scenario"),
    ],
    "Artificial Intelligence": [
        ("What is an intelligent agent designed to do?", ["Perceive and act toward goals", "Only store files", "Compile CSS", "Replace a database"], 0, "Agents perceive their environment and act toward goals.", "Agents", "conceptual"),
        ("Which approach represents knowledge with rules?", ["Symbolic AI", "Pixel caching", "Packet switching", "File compression"], 0, "Symbolic AI uses explicit symbols and rules.", "Knowledge", "comparison"),
        ("What is a heuristic?", ["A rule of thumb for search", "A database table", "A CPU register", "A CSS selector"], 0, "A heuristic estimates which choice may lead to a solution.", "Search", "terminology"),
        ("What does computer vision analyze?", ["Images or video", "Only SQL rows", "Audio cables", "Operating-system permissions"], 0, "Computer vision processes visual data.", "Perception", "application"),
        ("Why is evaluation important in an AI system?", ["It measures behavior against goals", "It removes all data", "It guarantees consciousness", "It changes the CPU"], 0, "Evaluation checks whether system behavior meets the intended objective.", "Evaluation", "reasoning"),
    ],
    "Machine Learning": [
        ("Which task predicts a continuous value?", ["Classification", "Regression", "Clustering", "Tokenization"], 1, "Regression predicts numeric values.", "Learning types", "conceptual"),
        ("Which method groups unlabeled examples?", ["Regression", "Clustering", "Classification", "Parsing"], 1, "Clustering discovers groups without labels.", "Unsupervised learning", "comparison"),
        ("What does a validation set help estimate?", ["Generalization", "Keyboard speed", "File size", "Syntax"], 0, "Validation estimates performance on unseen data during development.", "Evaluation", "application"),
        ("What is an embedding?", ["A numeric representation", "A database lock", "A compiler error", "A network cable"], 0, "Embeddings represent items as numeric vectors.", "Representations", "terminology"),
        ("What does overfitting mean?", ["Memorizing training patterns", "Using no data", "Sorting too fast", "Deleting labels"], 0, "An overfit model performs well on training data but poorly on new data.", "Generalization", "reasoning"),
    ],
    "Cloud Computing": [
        ("What does elasticity mean in cloud computing?", ["Resources scale with demand", "Data becomes encrypted automatically", "A CPU changes language", "A database loses indexes"], 0, "Elasticity adjusts resources as demand changes.", "Scaling", "conceptual"),
        ("Which model provides virtual machines over the internet?", ["IaaS", "SaaS", "DNS", "HTML"], 0, "Infrastructure as a service provides virtualized infrastructure.", "Service models", "terminology"),
        ("What is a region in a cloud platform?", ["A geographic deployment area", "A password group", "A code block", "A database column"], 0, "A region is a geographic area containing cloud infrastructure.", "Architecture", "application"),
        ("Which practice reduces dependence on one cloud vendor?", ["Portability planning", "Hard-coded proprietary APIs only", "Single-zone deployment", "Manual passwords"], 0, "Portability planning makes workloads easier to move.", "Architecture", "scenario"),
        ("What does serverless primarily hide from the developer?", ["Server management", "All application logic", "The user interface", "The data model"], 0, "Serverless platforms manage server provisioning and scaling.", "Service models", "comparison"),
    ],
}

VARIANT_PREFIXES = [
    "Core concept: ",
    "In a practical debugging review, ",
    "When choosing a design for a beginner project, ",
    "During an exam scenario, ",
]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def fingerprint_question(question: str, options: list[str]) -> str:
    value = normalize_text(question) + "|" + "|".join(normalize_text(option) for option in options)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def question_similarity(left: str, right: str) -> float:
    left_words = set(normalize_text(left).split())
    right_words = set(normalize_text(right).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def validate_question(item: dict[str, Any], subject: str, topic: str, subtopic: str, difficulty: str) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    options = [str(option).strip() for option in item.get("options", [])]
    requested_answer = item.get("answer", item.get("correct_answer", -1))
    if isinstance(requested_answer, str):
        answer_text = normalize_text(requested_answer)
        answer = next((index for index, option in enumerate(options) if normalize_text(option) == answer_text), -1)
    else:
        try:
            answer = int(requested_answer)
        except (TypeError, ValueError):
            answer = -1
    explanation = str(item.get("explanation") or "").strip()
    question_type = normalize_text(str(item.get("question_type") or "conceptual")).replace(" ", "-")
    normalized_difficulty = normalize_text(str(item.get("difficulty") or difficulty)).replace(" ", "-")
    normalized_difficulty = {"difficult": "hard", "beginner": "easy", "intermediate": "medium"}.get(normalized_difficulty, normalized_difficulty)
    if not question or not explanation or len(options) != 4 or any(not option for option in options):
        return None
    if (len({normalize_text(option) for option in options}) != 4 or answer not in range(4) or
            normalized_difficulty not in {"easy", "medium", "hard"} or question_type not in QUESTION_TYPES):
        return None
    return {
        "id": str(item.get("id") or ""),
        "question": question,
        "options": options,
        "answer": answer,
        "correct_answer": answer,
        "explanation": str(item.get("explanation") or "Review the concept and try again."),
        "subject": subject,
        "topic": topic,
        "subtopic": subtopic or str(item.get("subtopic") or "General"),
        "difficulty": normalized_difficulty,
        "fingerprint": fingerprint_question(question, options),
        "provider": str(item.get("provider") or "fallback"),
        "question_type": question_type,
    }


class AIProvider(ABC):
    name = "provider"

    @abstractmethod
    async def generate_questions(self, subject: str, topic: str, subtopic: str, difficulty: str, count: int, context: dict[str, Any]) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    @abstractmethod
    async def chat(self, message: str, context: dict[str, Any]) -> str | None:
        raise NotImplementedError


class FallbackProvider(AIProvider):
    name = "fallback"

    async def generate_questions(self, subject, topic, subtopic, difficulty, count, context):
        concepts = CONCEPTS.get(subject, CONCEPTS["Python"])
        excluded = set(context.get("excluded_fingerprints", []))
        output = []
        seen_types = set(context.get("recent_question_types", []))
        for variant_index in range(4):
            for concept_index, concept in enumerate(concepts):
                if len(output) >= count:
                    return output
                question, options, answer, explanation, concept_topic = concept[:5]
                question_type = (concept[-1] if len(concept) == 6 else ["conceptual", "scenario", "comparison", "reasoning"][variant_index])
                if variant_index and question_type in seen_types and len(seen_types) < 4:
                    question_type = ["conceptual", "scenario", "comparison", "reasoning"][variant_index]
                prompt = question if variant_index == 0 else VARIANT_PREFIXES[variant_index] + question[0].lower() + question[1:]
                item = validate_question({"id": f"fallback-{subject}-{variant_index}-{concept_index}", "question": prompt, "options": options, "answer": answer, "explanation": explanation, "subtopic": concept_topic, "provider": self.name, "question_type": question_type}, subject, topic or subject, subtopic, difficulty)
                if item and item["fingerprint"] not in excluded and item["fingerprint"] not in {q["fingerprint"] for q in output}:
                    output.append(item)
        return output

    async def chat(self, message, context):
        lowered = message.lower()
        topic = context.get("topic") or "your current topic"
        mastery = context.get("mastery", {}).get(topic)
        weak_topics = context.get("weak_topics", [])
        conversation = " ".join(item.get("content", "") for item in context.get("conversation", []))
        searchable = f"{lowered} {conversation.lower()}"
        concepts = CONCEPTS.get(topic, [])
        concept = next((item for item in concepts if any(term in searchable for term in item[4].lower().split())), None)
        if concept is None:
            concept = next((item for subject in SUBJECTS for item in CONCEPTS.get(subject, []) if item[4].lower() in searchable), None)
        if "what should i study" in lowered or "where should i start" in lowered or "next" in lowered:
            priority = weak_topics[0] if weak_topics else topic
            score = context.get("mastery", {}).get(priority)
            return f"Your best next focus is {priority}" + (f" at {score}% mastery" if score is not None else "") + ". Start with one short explanation, then do a small practice set and review every mistake before increasing difficulty."
        if "hint" in lowered:
            if concept:
                return f"Hint for {concept[4]}: focus on the defining rule in the question, then eliminate options that belong to a different concept. I will not reveal the answer before you attempt it."
            return f"Hint for {topic}: identify the core concept first, then eliminate options that describe a different layer or operation. I will not reveal the answer before you attempt it."
        if "why" in lowered and "wrong" in lowered:
            mistakes = context.get("recent_mistakes", [])
            detail = f" Your recent incorrect areas include {', '.join(mistakes[:2])}." if mistakes else ""
            return f"Let's inspect the mistake from {concept[4] if concept else topic}.{detail} Compare your choice with the defining rule, then check which condition the question is testing. Share the exact option for a precise explanation."
        if "test me" in lowered or "ask me" in lowered or "mcq" in lowered:
            return f"I can start a {topic} practice set now. I will use your recent mastery ({mastery if mastery is not None else 'not measured'}%) and avoid questions already shown to you."
        if "example" in lowered:
            if concept:
                return f"Example for {concept[4]}: {concept[0]} The key idea is {concept[3]} Try creating a small example with one changed input and predict what happens before running it."
            return f"Here is a compact example for {topic}: state the input, apply one rule at a time, and verify the output against the rule. Tell me the exact concept if you want a code or database example."
        if "compare" in lowered or "difference" in lowered:
            return f"To compare concepts in {topic}, write down their purpose, input/output behavior, and one trade-off. For the current concept, the defining rule is {concept[3] if concept else 'the rule that distinguishes it from nearby alternatives.'}"
        if "summar" in lowered or "recap" in lowered:
            summary = concept[3] if concept else f"{topic} is best learned by connecting definitions to worked examples and mistakes."
            return f"Quick summary of {concept[4] if concept else topic}: {summary} Next, explain it in your own words and solve one new practice question without looking at the answer."
        if "step" in lowered or "how do i" in lowered or "how to" in lowered:
            return f"Step-by-step for {concept[4] if concept else topic}: 1) identify the inputs, 2) state the rule, 3) apply one operation at a time, 4) check the result against an edge case, and 5) explain why the alternatives do not fit."
        if "simply" in lowered or "simple" in lowered:
            return f"In simple terms, {concept[4] if concept else topic} means {concept[3].lower() if concept else 'using a clear rule to turn a problem into a reliable result.'} We can build from a small example and increase difficulty as your answers improve."
        if concept:
            return f"{concept[4]}: {concept[3]} Ask me for a simpler explanation, a step-by-step walkthrough, an example, a hint, or a practice question."
        return f"I am your AI tutor for {topic}. Ask me to explain a concept, simplify it, compare ideas, summarize a topic, give an example, provide a hint, or say 'test me' to start adaptive practice."


class OpenAICompatibleProvider(AIProvider):
    def __init__(self, name: str, base_url: str, api_key: str = "", model: str = ""):
        self.name, self.base_url, self.api_key, self.model = name, base_url.rstrip("/"), api_key, model

    async def _request(self, system: str, prompt: str) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0.7}
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def generate_questions(self, subject, topic, subtopic, difficulty, count, context):
        prompt = f"Generate {count} distinct MCQs as a JSON array for {subject}, topic {topic}, subtopic {subtopic}, difficulty {difficulty}. Include question, options (exactly 4), answer (0-3), explanation, subtopic. Avoid these fingerprints: {context.get('excluded_fingerprints', [])}."
        try:
            content = await self._request("You are a careful educational assessment generator. Return JSON only.", prompt)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.I).strip())
            if isinstance(data, dict):
                data = data.get("questions", [])
            return [item for item in (validate_question(raw, subject, topic, subtopic, difficulty) for raw in data) if item]
        except Exception:
            return None

    async def chat(self, message, context):
        prompt = json.dumps({"learner_context": context, "message": message})
        try:
            return await self._request("You are a precise, encouraging tutor. Explain uncertainty, do not reveal quiz answers before an attempt, and stay on the learner's topic.", prompt)
        except Exception:
            return None


class GeminiProvider(AIProvider):
    """Gemini REST provider using the existing httpx dependency."""
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model or "gemini-2.5-flash"
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")

    def _url(self):
        return f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

    @staticmethod
    def _response_text(response):
        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(str(part.get("text", "")) for part in parts).strip()

    async def _generate(self, prompt, response_schema=None, json_mode=True):
        generation_config = {"temperature": 0.65}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        if response_schema:
            generation_config["responseSchema"] = response_schema
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": "You are a careful educational tutor. Return valid JSON when requested and never invent an answer key."}]},
            "generationConfig": generation_config,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self._url(), json=payload)
            response.raise_for_status()
            return self._response_text(response.json())

    async def generate_questions(self, subject, topic, subtopic, difficulty, count, context):
        if not self.api_key:
            return None
        previous = context.get("excluded_questions", [])[-30:]
        prompt = f"""Generate exactly {count} genuinely new MCQs.
Subject: {subject}
Topic: {topic}
Subtopic: {subtopic or 'choose an appropriate subtopic'}
Difficulty: {difficulty}
Learner mastery: {context.get('mastery', {})}
Weak topics: {context.get('weak_topics', [])}
Previously used questions, which must not be repeated, paraphrased, or reused as the same scenario:
{chr(10).join('- ' + item for item in previous) or '- none'}
Use different concepts and appropriate types such as conceptual, code-output, debugging, scenario, comparison, reasoning, terminology, practical, application, and problem-solving. Return only the requested JSON object."""
        schema = {"type": "OBJECT", "properties": {"questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"question": {"type": "STRING"}, "options": {"type": "ARRAY", "items": {"type": "STRING"}}, "correct_answer": {"type": "STRING"}, "explanation": {"type": "STRING"}, "difficulty": {"type": "STRING"}, "question_type": {"type": "STRING"}, "subject": {"type": "STRING"}, "topic": {"type": "STRING"}, "subtopic": {"type": "STRING"}}, "required": ["question", "options", "correct_answer", "explanation", "difficulty", "question_type", "subject", "topic", "subtopic"]}}}, "required": ["questions"]}
        try:
            content = await self._generate(prompt, schema)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.I).strip())
            raw_questions = data.get("questions", []) if isinstance(data, dict) else data
            result = []
            for raw in raw_questions:
                item = dict(raw)
                item["answer"] = item.get("correct_answer", item.get("answer", -1))
                validated = validate_question(item, subject, topic, subtopic, difficulty)
                if validated:
                    result.append(validated)
            return result
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    async def chat(self, message, context):
        if not self.api_key:
            return None
        prompt = json.dumps({"learner_context": context, "message": message}, ensure_ascii=True)
        try:
            return await self._generate("Respond as a patient educational tutor. Use the learner context, continue the current topic, explain uncertainty, and do not reveal quiz answers before an attempt. Return plain text, not JSON.\n" + prompt, None, False)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("ollama", settings.ollama_base_url.rstrip("/") + "/v1/chat/completions", "ollama", settings.ollama_model)


class HuggingFaceProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("huggingface", settings.hf_base_url, settings.hf_api_key, settings.hf_model)


def get_provider() -> AIProvider:
    if settings.gemini_api_key:
        return GeminiProvider()
    if settings.ai_provider == "ollama" and settings.ollama_base_url and settings.ollama_model:
        return OllamaProvider()
    if settings.ai_provider == "huggingface" and settings.hf_base_url and settings.hf_api_key and settings.hf_model:
        return HuggingFaceProvider()
    if settings.llm_base_url and settings.llm_model:
        return OpenAICompatibleProvider("llm", settings.llm_base_url + "/chat/completions", settings.llm_api_key, settings.llm_model)
    return FallbackProvider()
