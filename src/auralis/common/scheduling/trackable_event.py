import asyncio
import weakref
from typing import Set, Optional


class TrackableEvent:
    """
    An asyncio Event that tracks all its copies and allows checking if all related events are set.

    This event behaves like a standard asyncio.Event but maintains awareness of all its copies
    created through copy or deepcopy operations. Each event belongs to a "family" and can check
    if all events in its family are set.

    Example:
        event1 = TrackableEvent()
        event2 = copy.deepcopy(event1)

        event1.set()
        print(event2.are_all_set())  # False
        event2.set()
        print(event1.are_all_set())  # True
    """

    def __init__(self, parent=None):
        """
        Initialize a new TrackableEvent.

        Args:
            parent (Optional[TrackableEvent]): Parent event to inherit family from.
                If None, starts a new family. If provided, joins parent's family.
        """
        self._event = asyncio.Event()
        self._family: Set[weakref.ref] = set()

        if parent is None:
            # Start a new family
            self._family.add(weakref.ref(self, self._cleanup))
        else:
            # Clone, inherit the family
            self._family = parent._family
            self._family.add(weakref.ref(self, self._cleanup))

    def _cleanup(self, weak_ref):
        """
        Remove the weakref from family when the referenced object is garbage collected.

        Args:
            weak_ref: Weak reference to be removed
        """
        self._family.discard(weak_ref)

    def __copy__(self):
        """
        Create a shallow copy of the event that belongs to the same family.

        Returns:
            TrackableEvent: A new event instance in the same family
        """
        return self.__class__(parent=self)

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the event that belongs to the same family.

        Args:
            memo: Dictionary of already copied objects

        Returns:
            TrackableEvent: A new event instance in the same family
        """
        return self.__class__(parent=self)

    def set(self):
        """Set the event."""
        self._event.set()

    def clear(self):
        """Clear the event."""
        self._event.clear()

    async def wait(self):
        """
        Wait until the event is set.

        Returns:
            None: When the event is set
        """
        await self._event.wait()

    def is_set(self) -> bool:
        """
        Return True if and only if the event is set.

        Returns:
            bool: True if the event is set, False otherwise
        """
        return self._event.is_set()

    def are_all_set(self) -> bool:
        """
        Check if all events in the family are set.

        Returns:
            bool: True if all related events (including self) are set, False otherwise
        """
        return all(ref().is_set() for ref in self._family if ref() is not None)