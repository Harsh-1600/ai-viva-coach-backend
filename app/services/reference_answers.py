"""
Authoritative Technical Reference Answer Knowledge Base & Generator.
Provides technically accurate, pedagogical model answers across all core Computer Science domains:
1. Direct Answer
2. Simple Explanation (Real-world analogy)
3. Detailed Explanation (Internal architecture, mechanisms, and trade-offs)
4. Practical Example & Code (Line-by-line breakdown)
5. Key Points to Remember (Viva revision takeaways)
6. Improved Viva Answer (Model articulation for full marks)
"""

import re
from typing import Dict, Any, Optional, List


REFERENCE_DATABASE: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. Operating Systems: Virtual Memory, Paging, Page Faults, TLB
    # --------------------------------------------------------------------------
    "virtual_memory": {
        "correct_answer": (
            "Virtual memory is a memory management architecture that decouples a program's logical address space from physical RAM. "
            "The CPU Memory Management Unit (MMU) uses page tables and a Translation Lookaside Buffer (TLB) to translate virtual page numbers (VPN) "
            "to physical frame numbers (PFN). When a process accesses an address whose page is not resident in RAM (valid bit = 0 in the page table entry), "
            "the hardware triggers a page fault exception. The operating system kernel traps the fault, reads the missing page from disk swap storage into an "
            "available RAM frame, updates the page table entry, and transparently restarts the faulting instruction."
        ),
        "simple_explanation": (
            "Think of virtual memory like an index in a library. Instead of keeping every single book on your small study desk (RAM), "
            "you keep a catalog (page table). You only fetch the exact books you need right now. When your desk gets too full, "
            "the system swaps inactive books back to storage shelves (disk) so active work never crashes from running out of space."
        ),
        "detailed_explanation": (
            "1. Hardware Address Translation: Logical addresses are split into a Virtual Page Number (VPN) and an offset. The MMU consults the "
            "TLB cache. On a TLB hit, the Physical Frame Number (PFN) is concatenated with the offset in a single clock cycle. On a miss, "
            "the MMU walks a multi-level page table (e.g. 4-level paging in x86-64 using the CR3 register).\n"
            "2. Page Fault Interrupt Traps: If the PTE valid bit is 0, interrupt vector 14 traps execution to the kernel page fault handler. "
            "The OS selects a victim frame using replacement policies (LRU or CLOCK), writes dirty pages to swap, loads the requested page, "
            "updates permissions, and invalidates stale TLB entries across cores via TLB shootdowns.\n"
            "3. Memory Isolation & Protection: Each process has an independent page table hierarchy, preventing unauthorized cross-process memory access."
        ),
        "example": (
            "```c\n"
            "// In Linux, demand paging commits physical frames only on first access:\n"
            "char *buffer = (char *)malloc(1024 * 1024); // Reserves 1MB virtual address space; 0 RAM frames committed\n"
            "buffer[0] = 'A'; // Hardware triggers minor page fault; OS commits first 4KB physical frame on the fly\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 2 (`malloc`):** Reserves contiguous virtual address space but defers physical RAM allocation (demand paging).\n"
            "- **Line 3 (`buffer[0] = 'A'`):** First memory write raises a page fault trap; kernel allocates physical frame and updates PTE valid bit."
        ),
        "key_points_to_remember": [
            "Virtual memory isolates processes into independent address spaces and protects kernel memory.",
            "The Memory Management Unit (MMU) translates virtual addresses to physical frames using page tables and TLB caches.",
            "A page fault is an architectural trap raised when a virtual page is not resident in physical RAM.",
            "Standard page size is 4KB, balancing internal fragmentation against page table entry overhead."
        ],
        "improved_answer": (
            "\"Virtual memory decouples a process's logical address space from physical RAM, providing each process with an isolated, contiguous memory abstraction. "
            "Address translation is performed by the CPU MMU using page tables, heavily accelerated by the hardware Translation Lookaside Buffer (TLB). "
            "If an accessed page is not resident in physical memory (valid bit = 0), the MMU raises a page fault exception. The OS kernel traps the fault, "
            "allocates a physical frame—evicting a victim page to swap via LRU if memory is full—fetches the missing page from secondary storage, updates the page table, "
            "and transparently resumes the instruction without application interruption.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 2. Operating Systems: Deadlocks & Coffman Conditions
    # --------------------------------------------------------------------------
    "deadlock": {
        "correct_answer": (
            "A deadlock in Operating Systems is a state where a set of concurrent processes are permanently blocked because each process holds at least "
            "one resource and waits to acquire another resource held by another process in the set, creating a circular dependency that cannot resolve on its own. "
            "A deadlock can occur if and only if all four Coffman conditions hold simultaneously: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait."
        ),
        "simple_explanation": (
            "Imagine two cars driving from opposite directions onto a narrow single-lane bridge: neither car can move forward unless the other backs up, "
            "but neither driver is willing to reverse. Both cars remain permanently stuck at a complete standstill."
        ),
        "detailed_explanation": (
            "A deadlock strictly requires all four Coffman conditions:\n"
            "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
            "2. Hold and Wait: A process holding resources must simultaneously request additional resources held by others.\n"
            "3. No Preemption: Resources cannot be forcibly seized; they are released only voluntarily by the holding process.\n"
            "4. Circular Wait: A closed chain of processes P[0] -> P[1] -> ... -> P[n] exists where P[i] waits for a resource held by P[(i+1)%n].\n\n"
            "Remediation Strategies: Deadlock Prevention breaks at least one Coffman condition (e.g. enforcing strict global resource acquisition ordering to prevent circular wait). "
            "Deadlock Avoidance uses Banker's Algorithm to verify whether granting a resource request maintains a safe execution sequence."
        ),
        "example": (
            "```c\n"
            "// Thread 1 acquires Mutex A then Mutex B\n"
            "pthread_mutex_lock(&mutexA);\n"
            "pthread_mutex_lock(&mutexB); // Blocks if Thread 2 holds Mutex B\n\n"
            "// Thread 2 acquires Mutex B then Mutex A (Circular Wait Deadlock!)\n"
            "pthread_mutex_lock(&mutexB);\n"
            "pthread_mutex_lock(&mutexA); // Blocks if Thread 1 holds Mutex A\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- If Thread 1 locks `mutexA` while Thread 2 locks `mutexB`, neither thread can acquire its second lock, producing a circular wait deadlock.\n"
            "- **Fix:** Enforce lock hierarchy—both threads must acquire `mutexA` before `mutexB`."
        ),
        "key_points_to_remember": [
            "All four Coffman conditions (Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait) must hold simultaneously.",
            "Deadlock prevention invalidates at least one Coffman condition (commonly circular wait via strict resource ordering).",
            "Banker's Algorithm avoids deadlocks by simulating maximum resource claims to ensure the system remains in a safe state.",
            "Deadlock detection constructs a Resource Allocation Graph (RAG) and checks for cycles."
        ],
        "improved_answer": (
            "\"A deadlock is an execution standstill where a set of processes are permanently blocked waiting for resources held by each other. "
            "It occurs if and only if all four Coffman conditions hold concurrently: mutual exclusion, hold and wait, no preemption, and circular wait. "
            "To handle deadlocks in practice, systems employ prevention (breaking circular wait by acquiring locks in a strict global numerical order), "
            "avoidance (using Dijkstra's Banker's Algorithm to only grant allocations that keep the system in a safe state), or detection and recovery "
            "(periodically inspecting the Resource Allocation Graph for directed cycles and aborting victim processes).\""
        ),
    },

    # --------------------------------------------------------------------------
    # 3. Concurrency & Synchronization: Mutex vs Semaphore, Race Conditions
    # --------------------------------------------------------------------------
    "mutex_synchronization": {
        "correct_answer": (
            "A mutex (mutual exclusion lock) is a binary synchronization primitive used in concurrent programming to guarantee that only one thread "
            "can enter and execute a critical section at any given time, preventing race conditions and shared data corruption. A mutex possesses strict ownership "
            "semantics—only the thread that locked the mutex is permitted to unlock it. In contrast, a counting semaphore maintains an integer counter allowing "
            "up to N threads concurrent access and can be signaled across different threads."
        ),
        "simple_explanation": (
            "Think of a mutex like the single key to a private fitting room: only the person holding the key can enter and change clothes. "
            "Anyone else must wait outside until the occupant unlocks the door and hands back the key. A semaphore is like a parking lot with an electronic sign "
            "displaying 5 available spots; each entering car decrements the count until it reaches 0."
        ),
        "detailed_explanation": (
            "1. Atomic Primitives: Mutexes rely on hardware atomic instructions like Compare-And-Swap (CAS) or Test-and-Set to ensure race-free state transitions.\n"
            "2. OS Scheduler Integration: When contention occurs, waiting threads are suspended into a kernel wait queue, avoiding CPU-wasting busy-waits (spinlocks).\n"
            "3. Mutex vs Semaphore:\n"
            "   - Mutex: Binary locking (0 or 1), strict ownership semantics, priority inheritance support to combat priority inversion.\n"
            "   - Semaphore: Integer counter, signaling mechanism (Wait/Signal or P/V), no ownership restriction (can be unlocked by any thread)."
        ),
        "example": (
            "```python\n"
            "import threading\n\n"
            "lock = threading.Lock()\n"
            "shared_balance = 100\n\n"
            "def withdraw(amount):\n"
            "    global shared_balance\n"
            "    with lock:  # Critical section: acquires mutex, executes atomically, and releases\n"
            "        if shared_balance >= amount:\n"
            "            shared_balance -= amount\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 7 (`with lock`):** Acquires the mutex before reading or modifying `shared_balance`.\n"
            "- **Lines 8-9:** Prevents race conditions where two threads read the same balance concurrently and perform conflicting updates."
        ),
        "key_points_to_remember": [
            "A mutex guarantees mutual exclusion by allowing only one thread in a critical section at a time.",
            "Possesses strict ownership semantics—only the locking thread is permitted to unlock it.",
            "Implemented using atomic CPU primitives (CAS) combined with OS scheduler wait queues.",
            "To prevent deadlocks, multiple mutexes must always be acquired in a globally consistent order."
        ],
        "improved_answer": (
            "\"A mutex is a mutual exclusion synchronization primitive that protects critical sections by ensuring that only one thread accesses shared mutable state at a time. "
            "It is backed by hardware atomic operations like Compare-And-Swap (CAS) and integrates with the OS scheduler to sleep contending threads rather than burning CPU cycles. "
            "Crucially, a mutex features strict ownership—only the acquiring thread can release it, which also allows priority inheritance protocols to prevent priority inversion. "
            "This contrasts with counting semaphores, which manage permits for up to N concurrent threads and operate primarily as inter-thread signaling mechanisms.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 4. Databases: ACID Properties & Transactions
    # --------------------------------------------------------------------------
    "acid_transactions": {
        "correct_answer": (
            "ACID in database systems defines the four fundamental transaction properties guaranteeing data integrity despite hardware crashes, power outages, "
            "or concurrent access: 1) Atomicity ('all-or-nothing' execution; either all operations commit or the entire transaction rolls back); "
            "2) Consistency (transactions transform the database from one valid state to another, strictly enforcing all schema constraints, checks, and foreign keys); "
            "3) Isolation (concurrent transactions execute independently without observing uncommitted intermediate writes); and "
            "4) Durability (once committed, updates persist permanently in non-volatile storage even after a system crash)."
        ),
        "simple_explanation": (
            "Think of an ATM bank transfer: if $500 is debited from your checking account, it MUST be credited to your savings account. "
            "If the machine loses power halfway through, the transaction is reversed completely (Atomicity), money is neither created nor lost (Consistency), "
            "simultaneous withdrawals on the same account do not corrupt the balance (Isolation), and once you get your receipt, the update is permanent (Durability)."
        ),
        "detailed_explanation": (
            "1. Atomicity & Durability via Write-Ahead Logging (WAL): Changes are recorded sequentially in an append-only log on non-volatile disk before "
            "updating dirty data pages in memory. During crash recovery, the ARIES algorithm uses the log to REDO committed transactions and UNDO aborted ones.\n"
            "2. Isolation via Concurrency Control: Implemented via Strict Two-Phase Locking (S2PL) or Multi-Version Concurrency Control (MVCC) with snapshot isolation "
            "to prevent anomalies like dirty reads, non-repeatable reads, and phantom reads.\n"
            "3. Consistency: Enforced by transaction commit validation and database constraints (PRIMARY KEY, FOREIGN KEY, NOT NULL, CHECK)."
        ),
        "example": (
            "```sql\n"
            "BEGIN TRANSACTION;\n"
            "  UPDATE accounts SET balance = balance - 500 WHERE id = 1;\n"
            "  UPDATE accounts SET balance = balance + 500 WHERE id = 2;\n"
            "COMMIT; -- Guarantees atomic, durable persistence across both operations\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 1 (`BEGIN TRANSACTION`):** Establishes an isolation boundary.\n"
            "- **Lines 2-3:** Both account updates are held in memory and staged in the Write-Ahead Log (WAL).\n"
            "- **Line 4 (`COMMIT`):** Flushes WAL records to disk, making the transfer atomically and durably permanent."
        ),
        "key_points_to_remember": [
            "Atomicity guarantees 'all-or-nothing' execution via rollback logs or shadow paging.",
            "Consistency preserves relational integrity constraints and database invariants.",
            "Isolation prevents concurrent read/write anomalies using 2PL locking or MVCC snapshots.",
            "Durability ensures committed data survives system crashes via Write-Ahead Logging (WAL)."
        ],
        "improved_answer": (
            "\"ACID properties guarantee reliability and data integrity in database transactions. Atomicity ensures all operations succeed or the transaction rolls back completely. "
            "Consistency guarantees that transactions only transition the database from one valid state to another, upholding all constraints. "
            "Isolation prevents concurrent transactions from interfering with each other, implemented through Strict 2-Phase Locking or Multi-Version Concurrency Control (MVCC). "
            "Durability ensures that once committed, changes survive power outages and crashes via Write-Ahead Logging (WAL) and the ARIES recovery protocol.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 5. Databases: Indexing & B-Trees
    # --------------------------------------------------------------------------
    "database_indexing": {
        "correct_answer": (
            "A database index is a specialized auxiliary data structure—most commonly a B+ Tree—that significantly accelerates data retrieval operations "
            "on a database table at the cost of additional storage space and slower write operations (INSERT, UPDATE, DELETE). Instead of scanning every row "
            "in the table linearly (Full Table Scan, O(n)), the database engine traverses the balanced index tree in O(log n) disk I/O operations to locate "
            "the exact matching rows."
        ),
        "simple_explanation": (
            "Think of a database index like the index at the back of an 800-page textbook. Instead of reading through all 800 pages to find where 'Concurrency' "
            "is discussed, you flip straight to the alphabetical index, find the exact page number in seconds, and jump directly there."
        ),
        "detailed_explanation": (
            "1. B+ Tree Architecture: Unlike standard binary trees, B+ Trees have a high fan-out (hundreds of keys per node), keeping tree height low (typically 3-4 levels) "
            "to minimize expensive disk block reads. Internal nodes contain only routing keys; all actual data pointers/records reside in leaf nodes, which are linked sequentially.\n"
            "2. Clustered vs Non-Clustered Indexes:\n"
            "   - Clustered Index: Dictates the physical storage order of rows on disk (only one clustered index per table, usually the Primary Key).\n"
            "   - Non-Clustered (Secondary) Index: Separate structure containing index keys and row pointers (or clustered keys) back to the base table.\n"
            "3. Engineering Trade-Offs: While indexes speed up SELECT queries from O(n) to O(log n), every INSERT, UPDATE, and DELETE must also update index trees, "
            "introducing write amplification and node split overhead."
        ),
        "example": (
            "```sql\n"
            "-- Without index: EXPLAIN shows 'Full Table Scan' over 1,000,000 rows (O(n))\n"
            "SELECT * FROM users WHERE email = 'alice@example.com';\n\n"
            "-- Create B-tree index on lookup column:\n"
            "CREATE INDEX idx_users_email ON users(email);\n"
            "-- Now EXPLAIN shows 'Index Seek' using B+ Tree root-to-leaf traversal (O(log n))\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- Creating `idx_users_email` constructs a balanced B+ Tree ordered alphabetically by email.\n"
            "- Subsequent queries traverse 3-4 tree nodes instead of scanning millions of unindexed disk blocks."
        ),
        "key_points_to_remember": [
            "Indexes reduce query search time from linear scan O(n) to tree lookup O(log n).",
            "B+ Trees are preferred over B-Trees because sequential leaf node pointers accelerate range queries.",
            "Clustered index defines the physical order of table rows; non-clustered indexes store pointers.",
            "Trade-off: Indexes accelerate SELECT reads but add storage overhead and slow down INSERT/UPDATE/DELETE writes."
        ],
        "improved_answer": (
            "\"A database index is an auxiliary data structure, typically a B+ Tree, that optimizes read query performance by avoiding full table scans. "
            "B+ Trees maintain a high fan-out and shallow height (usually 3 to 4 levels), minimizing disk page reads. In a clustered index, table rows are physically "
            "sorted and stored in the leaf nodes, whereas a non-clustered index stores index keys alongside row locators. While indexes reduce search complexity from O(n) "
            "to O(log n) and optimize range scans through leaf-node pointer links, they require additional disk space and introduce write latency because mutations "
            "require tree rebalancing and node splits.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 6. Object-Oriented Programming: Polymorphism in Java/OOP
    # --------------------------------------------------------------------------
    "polymorphism": {
        "correct_answer": (
            "Polymorphism in Object-Oriented Programming is the principle that allows entities such as functions, objects, or interfaces to take on multiple forms. "
            "In Java and modern OOP, it manifests in two primary forms: 1) Compile-time (static) polymorphism, achieved through method overloading where multiple methods "
            "share the same name with different parameter signatures resolved at compile time; and 2) Runtime (dynamic) polymorphism, achieved through method overriding "
            "where a subclass provides a specific implementation of a superclass method, resolved dynamically at runtime via JVM dynamic method dispatch using virtual method tables (vtables)."
        ),
        "simple_explanation": (
            "Think of a universal remote control with a single 'Power' button: pressing that same button turns on a television, wakes up a stereo system, or toggles smart lights. "
            "The identical action triggers different, context-appropriate behaviors depending on the target device receiving the command."
        ),
        "detailed_explanation": (
            "1. Compile-Time Polymorphism (Static Binding): Method overloading and operator overloading. The compiler determines the exact method to bind during compilation "
            "based strictly on argument types and count. Calls are invoked directly via bytecode instructions like `invokestatic` or `invokespecial`.\n"
            "2. Runtime Polymorphism (Dynamic Binding): Method overriding and interface implementation. The method invocation `invokevirtual` uses dynamic dispatch. "
            "The JVM looks up the target object's Virtual Method Table (vtable) at runtime, resolving the method implementation associated with the concrete instance rather than the reference type.\n"
            "3. Architectural Significance: Polymorphism enables the Open/Closed Principle (OCP) and Dependency Inversion Principle (DIP), allowing software systems to be extended "
            "with new concrete classes without modifying existing caller code."
        ),
        "example": (
            "```java\n"
            "// Base class\n"
            "class Shape {\n"
            "    void draw() { System.out.println(\"Drawing a shape\"); }\n"
            "}\n"
            "// Derived class overriding base method\n"
            "class Circle extends Shape {\n"
            "    @Override\n"
            "    void draw() { System.out.println(\"Drawing a circle\"); }\n"
            "}\n"
            "public class Main {\n"
            "    public static void main(String[] args) {\n"
            "        Shape s = new Circle(); // Upcasting: Superclass reference holds subclass object\n"
            "        s.draw(); // Dynamic dispatch: JVM executes Circle's draw() at runtime via vtable\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 11 (`Shape s = new Circle()`):** Demonstrates upcasting—a superclass reference points to a concrete subclass object on the heap.\n"
            "- **Line 12 (`s.draw()`):** Resolved dynamically at runtime via the JVM vtable to invoke `Circle.draw()` instead of `Shape.draw()`."
        ),
        "key_points_to_remember": [
            "Polymorphism allows one interface or method call to control diverse concrete implementations.",
            "Method overloading is static binding (compile-time); method overriding is dynamic binding (runtime).",
            "In Java, dynamic method dispatch uses JVM virtual method tables (vtables) to resolve overridden methods.",
            "Static, private, and final methods cannot be overridden dynamically."
        ],
        "improved_answer": (
            "\"Polymorphism in OOP is the ability of an object or method invocation to exhibit multiple behaviors depending on its execution context. "
            "It is divided into compile-time polymorphism (achieved via method overloading, where the compiler resolves methods based on parameter signatures) "
            "and runtime polymorphism (achieved via method overriding and interface contracts). At runtime, when an overridden method is called on a superclass reference, "
            "the JVM uses dynamic method dispatch and inspects the concrete object's virtual method table (vtable) to invoke the subclass implementation. "
            "This decouples callers from concrete implementations, directly supporting the Open/Closed Principle.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 7. OOP: Encapsulation & Inheritance
    # --------------------------------------------------------------------------
    "oop_encapsulation_inheritance": {
        "correct_answer": (
            "Encapsulation and Inheritance are two foundational pillars of Object-Oriented Programming. Encapsulation is the bundling of data (attributes) "
            "and methods (behaviors) into a single unit (class) while restricting direct external access to internal state using access modifiers (private, protected, public), "
            "preserving internal state invariants. Inheritance is the mechanism by which a subclass derives fields and methods from an existing superclass, "
            "enabling code reusability, polymorphic substitution, and hierarchical domain modeling."
        ),
        "simple_explanation": (
            "Think of a modern automobile: the dashboard and pedals provide a clean public interface while complex engine components are contained under the hood (Encapsulation). "
            "Both sedans and electric trucks inherit common vehicle mechanics like wheels, braking, and steering from a base vehicle design (Inheritance)."
        ),
        "detailed_explanation": (
            "1. Encapsulation & Invariant Protection: Access modifiers safeguard class state from arbitrary external mutation. State modifications occur strictly "
            "through controlled public methods (getters, setters, business methods) that validate inputs before updating private fields.\n"
            "2. Inheritance & Liskov Substitution Principle (LSP): Subclasses inherit superclass state and behavior, extending or overriding functionality. "
            "According to LSP, objects of a superclass should be replaceable with objects of a subclass without breaking application correctness.\n"
            "3. Engineering Best Practice: Modern software architecture favors object composition over class inheritance ('Composition over Inheritance') "
            "to prevent tight coupling and brittle base class hierarchies."
        ),
        "example": (
            "```python\n"
            "class BankAccount:\n"
            "    def __init__(self, owner, balance):\n"
            "        self.owner = owner\n"
            "        self.__balance = balance  # Private attribute (Encapsulation)\n\n"
            "    def deposit(self, amount):\n"
            "        if amount > 0:            # Invariant validation\n"
            "            self.__balance += amount\n\n"
            "class SavingsAccount(BankAccount):    # Inheritance\n"
            "    def add_interest(self, rate):\n"
            "        # Extends BankAccount functionality while reusing base logic\n"
            "        pass\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 4 (`self.__balance`):** Private field prevents outside code from directly overwriting the account balance without validation.\n"
            "- **Line 10 (`class SavingsAccount(BankAccount)`):** Inherits base banking capabilities while allowing specialized savings features."
        ),
        "key_points_to_remember": [
            "Encapsulation binds data and behavior together and protects internal state invariants via access modifiers.",
            "Inheritance models 'IS-A' hierarchical relationships and enables code reusability.",
            "Subclasses must adhere to the Liskov Substitution Principle (LSP) to ensure predictable polymorphic behavior.",
            "Modern design principles favor composition over inheritance to avoid brittle, tightly coupled hierarchies."
        ],
        "improved_answer": (
            "\"Encapsulation and inheritance form the backbone of object-oriented architecture. Encapsulation enforces data hiding by restricting direct access "
            "to internal class attributes via access modifiers, ensuring state mutations only occur through validated public member functions. "
            "Inheritance establishes an IS-A relationship where child classes derive and extend properties from parent classes, facilitating code reuse. "
            "However, adhering to clean architecture requires ensuring subclasses satisfy the Liskov Substitution Principle, and software engineers "
            "frequently favor composition over inheritance to avoid deep, fragile inheritance trees.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 8. Data Structures: Binary Search Tree (BST) & Balanced Trees
    # --------------------------------------------------------------------------
    "binary_search_tree": {
        "correct_answer": (
            "A Binary Search Tree (BST) is a hierarchical node-based data structure where each node contains a key and at most two child nodes. "
            "It strictly enforces the binary search tree invariant: for any node N, all keys in its left subtree are strictly less than N.key, "
            "and all keys in its right subtree are strictly greater than N.key. This invariant guarantees that an in-order traversal (Left, Root, Right) "
            "visits all keys in sorted ascending order, and enables average-case O(log n) time complexity for search, insertion, and deletion operations."
        ),
        "simple_explanation": (
            "Think of a BST like searching for a word in a printed dictionary: opening directly to the middle allows you to instantly determine "
            "whether your word lies in the left half or right half, eliminating half the remaining book at each step."
        ),
        "detailed_explanation": (
            "1. Complexity Invariants: On a balanced BST of height h, search, insertion, and deletion operate in O(h) = O(log n) time. "
            "However, if keys are inserted in sorted order (e.g. 1, 2, 3, 4, 5), the tree degrades into an unbalanced skewed linked list with worst-case O(n) height.\n"
            "2. Self-Balancing Trees: To preserve O(log n) worst-case performance, self-balancing BST variants perform tree rotations upon node mutations:\n"
            "   - AVL Trees: Strictly enforce that the height difference (balance factor) between left and right subtrees is at most 1.\n"
            "   - Red-Black Trees: Use node coloring (Red/Black) and invariants to guarantee that the longest path from root to leaf is no more than twice the shortest path.\n"
            "3. Traversal Patterns: In-Order (L-Root-R) yields sorted order; Pre-Order (Root-L-R) is used for serialization; Post-Order (L-R-Root) is used for subtree deletion."
        ),
        "example": (
            "```python\n"
            "class Node:\n"
            "    def __init__(self, val):\n"
            "        self.val = val\n"
            "        self.left = None\n"
            "        self.right = None\n\n"
            "def search_bst(root, target):\n"
            "    if not root or root.val == target:\n"
            "        return root  # Found target or reached empty leaf\n"
            "    if target < root.val:\n"
            "        return search_bst(root.left, target)  # Recurse left\n"
            "    return search_bst(root.right, target)     # Recurse right\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 8 (`root.val == target`):** Base condition terminating search upon key match or reaching a null pointer.\n"
            "- **Lines 9-11:** Binary elimination halves search space at each recursive step based on the BST invariant."
        ),
        "key_points_to_remember": [
            "Enforces the BST ordering invariant: Left Subtree < Root < Right Subtree for all nodes.",
            "In-order traversal always outputs all keys in monotonically ascending sorted order.",
            "Average time complexity is O(log n); worst-case degrades to O(n) on pre-sorted data.",
            "Self-balancing trees (AVL / Red-Black) guarantee worst-case O(log n) operations via tree rotations."
        ],
        "improved_answer": (
            "\"A Binary Search Tree is a hierarchical node data structure adhering to the ordering property that for every node, all values in its left subtree "
            "are strictly smaller and all values in its right subtree are strictly larger. This invariant enables an in-order traversal to output keys in sorted order "
            "and supports O(log n) average search, insert, and delete operations. Because inserting sorted keys degrades an ordinary BST into an O(n) linear list, "
            "production systems use self-balancing BSTs such as AVL or Red-Black trees, which perform local pointer rotations to maintain bounded logarithmic height.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 9. Computer Networks: TCP vs UDP & 3-Way Handshake
    # --------------------------------------------------------------------------
    "tcp_udp_handshake": {
        "correct_answer": (
            "TCP (Transmission Control Protocol) is a connection-oriented, reliable transport-layer protocol providing ordered, error-checked, flow-controlled "
            "byte stream delivery. It establishes connections using a 3-Way Handshake (SYN, SYN-ACK, ACK) and implements sliding-window flow control and congestion control. "
            "In contrast, UDP (User Datagram Protocol) is a connectionless, lightweight, unreliable transport protocol that sends independent datagrams without handshake "
            "overhead, ordering guarantees, or retransmission, minimizing latency for real-time applications like VoIP, video streaming, and online gaming."
        ),
        "simple_explanation": (
            "Think of TCP like registered mail with signed delivery confirmations: every letter is tracked, numbered, and re-sent if lost in transit. "
            "UDP is like a live radio broadcast: transmissions are sent instantly in real-time, but if atmospheric interference causes you to miss a second of sound, "
            "the broadcast continues forward without replaying the missed audio."
        ),
        "detailed_explanation": (
            "1. The TCP 3-Way Handshake:\n"
            "   - Step 1 (SYN): Client sends a segment with SYN=1 and Initial Sequence Number ISN_C = x (enters SYN_SENT state).\n"
            "   - Step 2 (SYN-ACK): Server responds with SYN=1, ACK=1, its own ISN_S = y, and ack=x+1 (enters SYN_RCVD state).\n"
            "   - Step 3 (ACK): Client acknowledges with ACK=1, ack=y+1 (enters ESTABLISHED state). Server receives this and enters ESTABLISHED.\n"
            "2. Reliability & Ordering: TCP attaches 32-bit sequence and acknowledgment numbers to reconstruct out-of-order packets and retransmit lost segments.\n"
            "3. Flow & Congestion Control: Flow control uses a sliding receiver window (rwnd) to prevent overflowing the receiver buffer; congestion control (Slow Start, "
            "Congestion Avoidance, Fast Retransmit) prevents saturating network routers.\n"
            "4. Teardown: TCP connections are terminated gracefully using a 4-Way Handshake (FIN, ACK, FIN, ACK)."
        ),
        "example": (
            "```text\n"
            "Client [SYN_SENT]    ---- SYN (seq=100) -------------> Server [LISTEN -> SYN_RCVD]\n"
            "Client [ESTABLISHED] <--- SYN-ACK (seq=300, ack=101) - Server [SYN_RCVD]\n"
            "Client [ESTABLISHED] ---- ACK (ack=301) -------------> Server [ESTABLISHED]\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- Both peers independently select and synchronize 32-bit Initial Sequence Numbers.\n"
            "- A 3-way exchange is mathematically necessary to prevent delayed duplicate SYN packets from causing half-open connections."
        ),
        "key_points_to_remember": [
            "TCP is connection-oriented with guaranteed in-order delivery; UDP is connectionless with best-effort delivery.",
            "TCP connection establishment requires a 3-Way Handshake (SYN -> SYN-ACK -> ACK).",
            "TCP provides flow control via sliding windows and congestion control via AIMD/Slow Start.",
            "UDP has minimal header overhead (8 bytes vs 20+ bytes in TCP) and zero handshake latency."
        ],
        "improved_answer": (
            "\"TCP and UDP represent contrasting transport-layer trade-offs between reliability and latency. TCP is connection-oriented, establishing a synchronized "
            "duplex byte stream via a 3-way handshake (SYN, SYN-ACK, ACK) and tearing down via a 4-way FIN exchange. It guarantees ordered delivery, retransmits lost packets "
            "using sequence numbers, and manages network capacity through sliding-window flow control and congestion avoidance algorithms. UDP, by comparison, "
            "is connectionless and best-effort, transmitting datagrams with minimal 8-byte header overhead and no handshake latency, making it ideal for DNS lookups, "
            "VoIP, and real-time multiplayer gaming where timeliness outweighs absolute reliability.\""
        ),
    },

    # --------------------------------------------------------------------------
    # 10. Algorithms: QuickSort vs MergeSort
    # --------------------------------------------------------------------------
    "sorting_algorithms": {
        "correct_answer": (
            "QuickSort and MergeSort are two fundamental divide-and-conquer comparison sorting algorithms. QuickSort selects a pivot element, partitions "
            "the array so elements smaller than the pivot precede it and larger follow it, and recursively sorts sub-arrays in-place (average time complexity O(n log n), "
            "worst-case O(n^2), auxiliary space O(log n)). MergeSort divides the array into two halves, recursively sorts them, and merges the sorted halves "
            "using an auxiliary array, guaranteeing O(n log n) time complexity in all cases and stable sorting at the cost of O(n) auxiliary space."
        ),
        "simple_explanation": (
            "Think of sorting a large stack of exam papers alphabetically: QuickSort picks a reference name (pivot, say 'M') and divides papers into 'A-L' "
            "and 'N-Z' piles directly on the table. MergeSort splits the stack in half repeatedly until single papers remain, then zips pairs together in order."
        ),
        "detailed_explanation": (
            "1. QuickSort Mechanics: Partitioning (Lomuto or Hoare) places the pivot in its final sorted position in O(n) time. Average-case time is O(n log n) with "
            "exceptional cache locality. However, if the pivot is repeatedly the extreme minimum or maximum (e.g. on sorted input with naive pivot selection), "
            "recursion depth degrades to O(n), producing O(n^2) worst-case time.\n"
            "2. MergeSort Mechanics: The array is recursively halved until base size 1. The merge step uses two pointers to combine sorted sub-arrays into an auxiliary buffer. "
            "Because merging requires O(n) extra memory, MergeSort is not in-place, but it is guaranteed O(n log n) and stable (preserves relative order of duplicate keys).\n"
            "3. Engineering Choice: QuickSort is widely used for in-memory primitive sorting (e.g. C's `qsort`); MergeSort/Timsort is preferred for linked lists and object arrays."
        ),
        "example": (
            "```python\n"
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
            "```\n\n"
            "**Line-by-Line Breakdown:**\n"
            "- **Line 4 (`pivot = ...`):** Selects pivot element to partition remaining items.\n"
            "- **Lines 5-7:** Divides array into elements less than, equal to, and greater than pivot.\n"
            "- **Line 8:** Conquers subproblems recursively and concatenates results."
        ),
        "key_points_to_remember": [
            "QuickSort is in-place with average O(n log n) time and high cache performance, but has O(n^2) worst-case time.",
            "MergeSort is stable and guarantees O(n log n) time in all cases, but requires O(n) extra space.",
            "Pivot selection strategies (randomized, median-of-three) defend QuickSort against worst-case O(n^2) degradation.",
            "MergeSort is optimal for external sorting and linked lists where sequential access is dominant."
        ],
        "improved_answer": (
            "\"QuickSort and MergeSort are classic divide-and-conquer sorting algorithms that present distinct performance trade-offs. "
            "QuickSort partitions the array around a pivot in-place and recursively sorts sub-partitions. It achieves O(n log n) average time complexity "
            "with superior cache locality, though poor pivot selection can degrade it to O(n^2). Conversely, MergeSort guarantees strict O(n log n) "
            "time complexity across best, average, and worst cases, and preserves stability, but requires O(n) auxiliary memory to merge sub-arrays. "
            "Consequently, in-place QuickSort variants are favored for system primitives, while MergeSort derivatives like Timsort are standard for object collections.\""
        ),
    },
}


def match_reference_topic(question_text: str, topic: str = "") -> Optional[str]:
    """Matches a question and topic string to a canonical reference database key."""
    combined = f"{topic} {question_text}".lower()

    if any(k in combined for k in ["page", "paging", "virtual memory", "mmu", "tlb", "address translation"]):
        return "virtual_memory"
    if any(k in combined for k in ["deadlock", "coffman", "banker", "circular wait"]):
        return "deadlock"
    if any(k in combined for k in ["mutex", "semaphore", "synchroniz", "race condition", "critical section"]):
        return "mutex_synchronization"
    if any(k in combined for k in ["acid", "transaction", "durability", "isolation level", "wal"]):
        return "acid_transactions"
    if any(k in combined for k in ["index", "indexing", "b-tree", "b+ tree", "clustered index"]):
        return "database_indexing"
    if any(k in combined for k in ["polymorphism", "overriding", "overloading", "dynamic dispatch", "vtable"]):
        return "polymorphism"
    if any(k in combined for k in ["encapsulation", "inheritance", "abstraction", "oop", "object-oriented"]):
        return "oop_encapsulation_inheritance"
    if any(k in combined for k in ["binary search tree", "bst", "avl", "red-black", "tree traversal"]):
        return "binary_search_tree"
    if any(k in combined for k in ["tcp", "udp", "handshake", "socket", "transport layer"]):
        return "tcp_udp_handshake"
    if any(k in combined for k in ["quicksort", "mergesort", "sorting", "sort algorithm"]):
        return "sorting_algorithms"

    return None


def get_authoritative_reference_solution(
    question_text: str,
    topic: str = "",
    subject: str = "",
    difficulty: str = "Medium"
) -> Dict[str, Any]:
    """Retrieves an authoritative reference model answer, explanation, example, key points, and improved viva answer."""
    key = match_reference_topic(question_text, topic)
    if key and key in REFERENCE_DATABASE:
        ref = REFERENCE_DATABASE[key]
        return {
            "correct_answer": ref["correct_answer"],
            "simple_explanation": ref["simple_explanation"],
            "detailed_explanation": ref["detailed_explanation"],
            "example": ref["example"],
            "key_points_to_remember": list(ref["key_points_to_remember"]),
            "improved_answer": ref.get("improved_answer") or ref["correct_answer"],
            "is_matched": True,
        }

    # Dynamic fallback generator for any custom or specialized question
    clean_q = question_text.strip()
    clean_topic = (topic or subject or "Computer Science").strip()

    dynamic_ans = (
        f"In {clean_topic}, addressing '{clean_q}' requires understanding the fundamental architectural mechanisms, "
        "operational data flow, and practical engineering trade-offs. An authoritative technical answer defines the primary principles, "
        "details step-by-step component interactions, and articulates performance and system constraints."
    )
    dynamic_simple = (
        f"Think of {clean_topic} as an organized system contract: it establishes clear rules and interfaces so different software "
        "and hardware components can coordinate work reliably without data corruption or unexpected failures."
    )
    dynamic_detailed = (
        f"1. Core Principles: Explains the theoretical foundations, data structures, and state transitions relevant to '{clean_q}'.\n"
        f"2. Execution Workflow: Details the control flow, memory utilization, and failure recovery protocols in {clean_topic}.\n"
        f"3. Practical Trade-offs: Weighs computational time complexity against memory overhead, scalability, and system bottlenecks."
    )
    dynamic_example = (
        f"```python\n"
        f"# Practical implementation pattern for {clean_topic}\n"
        f"def process_operation(context):\n"
        f"    # Step 1: Precondition check protects system invariants\n"
        f"    if not context:\n"
        f"        raise ValueError('Invalid execution context')\n"
        f"    # Step 2: Execute core mechanism reliably\n"
        f"    return execute_step(context)\n"
        f"```"
    )
    dynamic_key_points = [
        f"State the formal technical definition of the concept at the outset of your viva response.",
        f"Detail the internal mechanisms, component state lifecycles, and edge case exceptions.",
        f"Always articulate engineering trade-offs (time vs space, throughput vs latency) to the examiner."
    ]
    dynamic_improved = (
        f"\"To answer '{clean_q}' comprehensively in a viva: Begin with a direct, unambiguous definition of the core concept in {clean_topic}. "
        "Next, explain the underlying mechanism step-by-step, including data flow and component interactions. Conclude by highlighting real-world "
        "trade-offs, failure modes, and practical performance considerations to demonstrate deep technical mastery.\""
    )

    return {
        "correct_answer": dynamic_ans,
        "simple_explanation": dynamic_simple,
        "detailed_explanation": dynamic_detailed,
        "example": dynamic_example,
        "key_points_to_remember": dynamic_key_points,
        "improved_answer": dynamic_improved,
        "is_matched": False,
    }
