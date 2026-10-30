"""
Alert Management and Escalation Subsystem
Manages tiered alerts (Visual, Audio Chime, Emergency Alarm) with hysteresis and cooldowns.
"""

import time
import logging
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger("AlertManager")


class AlertLevel(Enum):
    NORMAL = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL = 3


class AlertManager:
    """
    Tiered state-machine for triggering safety interventions without alert fatigue.
    """
    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        sound_player_callback: Optional[Callable[[str], None]] = None
    ):
        self.cooldown_seconds = cooldown_seconds
        self.sound_player_callback = sound_player_callback
        self.current_level = AlertLevel.NORMAL
        self.last_alert_time: float = 0.0
        self.total_alerts_triggered = 0

    def evaluate(self, status: str, fatigue_score: float) -> AlertLevel:
        """Evaluates detection status and determines alert level."""
        now = time.time()

        if status == "DROWSY" or fatigue_score >= 0.8:
            new_level = AlertLevel.CRITICAL
        elif status in ("DISTRACTED", "YAWN_WARNING") or fatigue_score >= 0.5:
            new_level = AlertLevel.WARNING
        elif status == "FATIGUE_ADVISORY" or fatigue_score >= 0.35:
            new_level = AlertLevel.ADVISORY
        else:
            new_level = AlertLevel.NORMAL

        # State transition handling with cooldown logic for audio alarms
        if new_level.value > AlertLevel.NORMAL.value:
            if (now - self.last_alert_time) >= self.cooldown_seconds:
                self.last_alert_time = now
                self.total_alerts_triggered += 1
                self._dispatch_alert(new_level, status)

        self.current_level = new_level
        return self.current_level

    def _dispatch_alert(self, level: AlertLevel, reason: str):
        """Dispatches audio and system alerts based on severity."""
        logger.warning(f"Safety Alert Dispatched: Level={level.name}, Reason={reason}")
        if self.sound_player_callback and level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            try:
                self.sound_player_callback(level.name)
            except Exception as e:
                logger.error(f"Failed to play alarm: {e}")
