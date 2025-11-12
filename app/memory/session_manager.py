"""
Session Manager - Persistent storage for conversation history and context
Uses SQLite for simple, local persistence
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session persistence and retrieval"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize session manager
        
        Args:
            db_path: Path to SQLite database (default: app/memory/sessions.sqlite)
        """
        if db_path is None:
            db_path = Path(__file__).parent / "sessions.sqlite"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"SessionManager initialized: {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                context TEXT,
                metadata TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                data TEXT,
                sql TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """
        Create a new session
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
            
        Returns:
            True if created, False if already exists
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO sessions (session_id, created_at, updated_at, context, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                now,
                now,
                json.dumps({}),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created session: {session_id}")
            return True
            
        except sqlite3.IntegrityError:
            logger.debug(f"Session already exists: {session_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, created_at, updated_at, context, metadata
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "session_id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2],
                    "context": json.loads(row[3]) if row[3] else {},
                    "metadata": json.loads(row[4]) if row[4] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def update_context(self, session_id: str, context: Dict):
        """
        Update session context
        
        Args:
            session_id: Session identifier
            context: Context dictionary to save
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                UPDATE sessions
                SET context = ?, updated_at = ?
                WHERE session_id = ?
            """, (json.dumps(context), now, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        data: Optional[Dict] = None,
        sql: Optional[str] = None
    ):
        """
        Add message to session
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            data: Optional result data
            sql: Optional SQL query
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO messages (session_id, timestamp, role, content, data, sql)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                now,
                role,
                content,
                json.dumps(data) if data else None,
                sql
            ))
            
            # Update session timestamp
            cursor.execute("""
                UPDATE sessions
                SET updated_at = ?
                WHERE session_id = ?
            """, (now, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get messages for a session
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of message dictionaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if limit:
                cursor.execute("""
                    SELECT timestamp, role, content, data, sql
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT timestamp, role, content, data, sql
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                messages.append({
                    "timestamp": row[0],
                    "role": row[1],
                    "content": row[2],
                    "data": json.loads(row[3]) if row[3] else None,
                    "sql": row[4]
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recently updated sessions
        
        Args:
            limit: Number of sessions to return
            
        Returns:
            List of session summaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, created_at, updated_at, metadata
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            sessions = []
            for row in rows:
                sessions.append({
                    "session_id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {}
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    def delete_session(self, session_id: str):
        """
        Delete a session and all its messages
        
        Args:
            session_id: Session identifier
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
    
    def cleanup_old_sessions(self, days: int = 30):
        """
        Delete sessions older than specified days
        
        Args:
            days: Number of days to keep
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get old session IDs
            cursor.execute("""
                SELECT session_id FROM sessions
                WHERE updated_at < ?
            """, (cutoff,))
            
            old_sessions = [row[0] for row in cursor.fetchall()]
            
            if old_sessions:
                # Delete messages
                placeholders = ','.join('?' * len(old_sessions))
                cursor.execute(f"""
                    DELETE FROM messages
                    WHERE session_id IN ({placeholders})
                """, old_sessions)
                
                # Delete sessions
                cursor.execute(f"""
                    DELETE FROM sessions
                    WHERE session_id IN ({placeholders})
                """, old_sessions)
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_sessions)} old sessions")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with stats
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM sessions
                WHERE updated_at > ?
            """, ((datetime.now() - timedelta(days=7)).isoformat(),))
            active_sessions = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "active_sessions_7d": active_sessions
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


