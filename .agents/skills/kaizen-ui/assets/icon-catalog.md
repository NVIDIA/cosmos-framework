# Icon catalog

Action→icon mapping and the full list of available icon names. This file is data — for whether/how to use icons (decision rules, variant selection, anti-patterns, icon-only safe set), see [references/ICONS.md](../references/ICONS.md).

Use the same icon for the same conceptual action across the app.

## Imports

Icons are named PascalCase exports from `@nvidia/react-gui-icons/icons` and accept a `variant="fill"` prop for the solid version. Names that would start with a digit or shadow a JS global are prefixed `Icon` (`IconError`, `IconImage`, `Icon3DAxis`).

```tsx
import { Add, Pencil, Trash, MagnifyingGlass } from "@nvidia/react-gui-icons/icons";
```

The same icons are also available from `@nvidia/react-gui-icons` as `<NvidiaGuiIcon iconName="kebab-case" />` — see [references/ICONS.md](../references/ICONS.md) Setup section.

---

## CRUD and document actions

| Action | Icon | Notes |
|---|---|---|
| Rename | `Rename` | |
| Archive | `Archive` | |
| Import | `Import` | |
| Export | `Export` | |
| Upload | `Upload` | |
| Download | `Download` | |

## Navigation and wayfinding

| Action | Icon | Notes |
|---|---|---|
| Back | `ArrowLeft` | |
| Forward | `ArrowRight` | |
| Navigate left/right | `ChevronLeft` / `ChevronRight` | Pagination, carousel, breadcrumb navigation |
| Home | `Home` | |
| Menu (hamburger) | `Menu` | Mobile/responsive menu trigger |

## View and display

| Action | Icon | Notes |
|---|---|---|
| Expand / Fullscreen | `Expand` | |
| Collapse / Exit fullscreen | `Collapse` | |
| Grid view | `LayoutGrid` | For SegmentedControl view toggles |
| List view | `LayoutList` | For SegmentedControl view toggles |
| Column view | `LayoutColumns` | |
| Show / Visible | `Eye` | |
| Hide / Hidden | `EyeOff` | |

## Status and feedback

| Action | Icon | Notes |
|---|---|---|
| Information | `InfoCircle` | Tooltips, info popovers, help |
| Help | `HelpCircle` | |
| Warning | `Warning` | Triangle alert |
| Error | `IconError` | |
| Success / Check | `Check` | Avoid `CheckCircle` in dense contexts — the enclosing circle adds visual weight at small sizes |
| Settings / Configure | `Cog` | |
| Notifications / Alerts | `Bell` | |

## User and account

| Action | Icon | Notes |
|---|---|---|
| User / Profile | `Profile` | Single user |
| User group | `ProfileGroup` | Multiple users |
| Add user | `AccountAdd` | |
| Remove user | `AccountRemove` | |
| Lock / Secure | `LockClosed` | |
| Unlock | `LockOpen` | |

## Content and media

| Action | Icon | Notes |
|---|---|---|
| Image | `IconImage` | |
| Document | `IconDocument` | |
| Folder (closed) | `FolderClosed` | |
| Folder (open) | `FolderOpen` | |
| Link | `Link` | |
| Unlink | `LinkBreak` | |
| Bookmark | `Bookmark` | Good candidate for solid/outline active state toggle |
| Star / Favorite | `Star` | Good candidate for solid/outline active state toggle |
| Pin | `Pin` | |

---

## All available icons

Reference list of every icon name the design system uses. These are design-system names, not the export list of any one package: with a different icon library, map each name to its closest equivalent.
For `NvidiaGuiIcon`, pass the kebab-case form of the name.

### 3d

- Cloth
- Cube
- Dolly
- Emissive
- Explode
- Icon3DAxis
- Instance
- LightRect
- LightRotate
- Material
- MoveGlobal
- Reference
- Rotate3D
- RotateGlobal
- Segment
- Shader
- ShaderBall
- Skeleton
- Teapot
- Teleport
- Texture

### Av

- AutoRecord
- AutoRecordOff
- Broadcast
- Camera
- CameraAdd
- CameraOff
- ClapBoard
- ClosedCaption
- DisplaySurround
- Eject
- Equalizer
- Faders
- FastForward
- FastReverse
- FileReel
- Film
- FilmAdd
- FilmCollection
- Forward10
- Fps
- HighQuality
- Hud
- Loop
- LoopOff
- Microphone
- MicrophoneOff
- Next
- Notes
- Pause
- PictureInPicture
- Play
- Previous
- Record
- Replay
- ReplayOff
- Reverse10
- Shuffle
- Soundwaves
- Speaker
- SpeakerHigh
- SpeakerLow
- SpeakerMedium
- SpeakerMute
- SpeakerOff
- StepForward
- StepReverse
- Stop
- Videocam
- VideocamOff
- WatchParty

### Common

- Add
- AddCircle
- Alarm
- Bell
- BellOff
- Calendar
- CalendarEvent
- Cancel
- Check
- CheckCircle
- CheckMultiCircle
- Clock
- Close
- CloseCircle
- Cog
- CogOff
- CopyDoc
- CopyGeneric
- Enter
- Exit
- Eye
- EyeOff
- Filter
- HelpCircle
- Home
- HomeOff
- IconClipboard
- IconError
- IconHistory
- Inbox
- InfoCircle
- Link
- LinkBreak
- LockClosed
- LockOpen
- MagnifyingGlass
- MagnifyingGlassMinus
- MagnifyingGlassPlus
- Menu
- MoreHoriz
- MoreVert
- Outbox
- Paperplane
- Redo
- Refresh
- Reset
- Retry
- Share
- Sort
- Star
- Subtract
- SubtractCircle
- Sync
- SyncOff
- Timer
- Trash
- Undo
- Warning

### Communication

- Cloud
- CloudOff
- Data
- Download
- Envelope
- EnvelopeOff
- Firewall
- FirewallOff
- Forward
- Isp
- IspOff
- NetworkSignal
- NetworkSignalOff
- Reply
- SyncWarning
- TransferHorizontal
- TransferVertical
- Upload
- Wifi
- WifiOff

### Cursor

- Arrow
- Crosshair
- HandClosed
- HandOpen
- HandPinch
- HandPoint
- Hourglass
- HourglassEnd
- HourglassMid
- HourglassStart

### Editor

- Anchor
- AnchorCenter
- AnchorE
- AnchorN
- AnchorNe
- AnchorNw
- AnchorS
- AnchorSe
- AnchorSw
- AnchorW
- Approximately
- ArrangeColumns
- ArrangeRows
- AtSymbol
- Backspace
- Bold
- BoundingBox
- Chart
- ChartBar
- ChartBubble
- ChartFlow
- ChartHierarchy
- ChartPerformance
- ChartPie
- ChartScatterplot
- ChartTree
- Collapse
- Commit
- Connection
- CraftKnife
- Divide
- Equal
- Expand
- ExportFromClipboard
- ExportToClipboard
- Fit
- Freehand
- Generate
- Grid
- GridOff
- HAlignCenter
- HAlignLeft
- HAlignRight
- Header
- Highlighter
- IconImage
- IconText
- Italic
- Iterate
- Keyframe
- Lasso
- Layers
- LayersOff
- LineSegment
- ListBullet
- ListCheckmark
- ListNumber
- Markdown
- Move
- NotEqual
- Numbers
- Paperclip
- Paragraph
- Pen
- Pencil
- Rename
- Rotate
- Rotate90Clockwise
- Rotate90Counter
- Ruler
- ScaleDown
- ScaleReset
- ScaleUp
- Scissors
- ScmBranch
- ScmCompare
- ScmFork
- ScmMerge
- ScmPull
- ScmPullClose
- ScmPullDraft
- Section
- SectionBottom
- SectionTop
- SelectBrush
- Signature
- Skip
- Sliders
- SortAscending
- SortDescending
- Sticker
- StickerImage
- StickerShape
- StickerText
- StickerZoom
- Strikethrough
- StrokeWidth
- Style
- Swatches
- TextAlignCenter
- TextAlignFull
- TextAlignLeft
- TextAlignRight
- TextFramed
- TrashDelete
- Typography
- Underline
- VAlignBottom
- VAlignCenter
- VAlignTop
- Wand
- Wrench

### Files

- Archive
- Db
- DocumentCheckmark
- DocumentNew
- DocumentPreview
- Export
- Floppy
- FolderClosed
- FolderOpen
- IconDocument
- Import
- SplitFile

### Hardware

- Battery0
- Battery0Off
- Battery100
- Battery25
- Battery50
- Battery75
- BatteryBoost
- BatteryCharging
- Bluetooth
- BluetoothOff
- Calculator
- Coaxial
- Component
- Cpu
- Display
- DisplayCollection
- DisplayColumn
- DisplayOff
- DisplayRow
- Displayport
- DriveCloud
- DriveCloudOff
- DriveNetwork
- DriveNetworkOff
- DriveRemovable
- DriveRemovableOff
- DriveUsb
- DriveUsbOff
- Drone
- Dvi
- Ethernet
- EthernetPort
- Fan
- FanLoud
- FanQuiet
- FlightStick
- Gamepad
- GamepadOff
- GamepadStart
- Gpu
- GpuCard
- GpuCardMulti
- GpuCardOff
- GpuOff
- GpuSync
- Hdmi
- Headset
- Icon2BLeftClick
- IconKeyboard
- JoystickClassic
- Laptop
- Laser
- Lightning
- Mac
- MacOff
- MediaOptical
- MiniDisplayport
- Mosaic
- Mouse2B
- Mouse2BRightClick
- Mouse3B
- MouseScrollwheel
- MouseScrollwheelActive
- Multidrive
- Network
- NetworkConnection
- NetworkPcs
- NvidiaShield
- NvidiaShieldStand
- OutletA
- OutletB
- OutletC
- OutletE
- OutletG
- OutletH
- OutletI
- OutletJ
- OutletK
- OutletL
- PerformanceHigh
- PerformanceLow
- PerformanceMedium
- PhoneLaptop
- PlugReceptacle
- PlugUsb
- Plugin
- Power
- PowerSupplyUnit
- Ram
- Remote
- Robot
- RobotArm
- RobotDelivery
- RobotTorso
- RobotWheels
- SdCard
- Smartphone
- SmartphoneOff
- Tablet
- TabletOff
- Telephone
- TelephoneOff
- Television
- TelevisionRetro
- Usb
- Vga
- Vr
- Watch
- Webcam
- WebcamOff
- WebcamSettings
- Whispermode
- WirelessModem
- WirelessModemOff
- Workstation
- WorkstationOff
- WorkstationSystem
- WorkstationSystemOff

### Image

- Brush
- Bucket
- Camera360
- CameraLinked
- CameraSuper
- Collection
- Crop
- Denoise
- Easel
- Eraser
- Eyedropper
- Group
- ImportImage
- MaskInside
- MaskOutside
- MirrorHorizontal
- MirrorVertical
- Outpainting
- Palette
- Scale
- ScaleRelative
- ScaleRelativeOff
- SelectEllipse
- SelectPolygon
- SelectRectangle
- SendImage
- Shapes
- SunHigh
- SunLow
- SunOffHigh
- Transparency

### Maps

- Airplane
- Airport
- Bank
- Bed
- Bicycle
- Briefcase
- Bush
- Business
- Castle
- City
- Datacenter
- Dirt
- Elevator
- Emergency
- Escalator
- Factory
- Farm
- Flag
- Flower
- Fog
- FuelCan
- FuelPump
- Government
- Grass
- Gravel
- Helicopter
- Hill
- IconLocation
- IconMap
- LocationOff
- MapOff
- Mountain
- Mud
- Park
- PhysicalExit
- River
- Rock
- Route
- Sailboat
- Sand
- Sea
- Ship
- Signpost
- Sky
- Snow
- Stair
- Stone
- Straw
- Suitcase
- Toolbox
- Tree
- Trees
- Water
- World
- WorldOff

### Misc

- Accessibility
- Atom
- Bandaid
- Barcode
- Beaker
- Blackboard
- Bone
- Book
- Bottle
- Braille
- Bug
- BugOff
- Bus
- Car
- Cards
- CheckmarkBadge
- Circle1Q
- Circle2Q
- Circle3Q
- CircleTick
- Code
- Compass
- Concierge
- Crown
- CubeStack
- DeliveryTruck
- Die1
- Die2
- Die3
- Die4
- Die5
- Die6
- Ekg
- Engine
- FacialRecognition
- Fingerprint
- Fire
- Firecracker
- Firework
- Food
- Footprint
- Fork
- Gift
- Graduate
- GraphNode
- GraphNodeConnect
- GraphNodeDelete
- Head
- Health
- HighKick
- Hit
- HotAirBalloon
- Hydrant
- IconFunction
- IconInfinity
- IconMath
- IconSensor
- Ime
- Key
- Knife
- Lamp
- Library
- LibraryGames
- Lifebuoy
- Lightbulb
- Lighthouse
- Magnet
- Man
- ManWoman
- Mapping
- Martini
- Meal
- Medicine
- Megaphone
- Microscope
- ModeComponent
- Money
- MoneyCoin
- MoneyPaper
- Moon
- MoonOff
- Mug
- MugBeer
- NeuralNetwork
- News
- Os
- Package
- Pawn
- Pawprint
- PaymentCard
- Pizza
- Placesetting
- Platformer
- Podium
- PrayerRoom
- Projector
- Pulse
- Puzzle
- Qrcode
- Radar
- Radioactive
- Rocket
- Running
- SawWave
- ScaleBalance
- School
- Secure
- Seed
- Shield
- ShieldOff
- ShipWheel
- ShoppingBag
- ShoppingBasket
- ShoppingCart
- ShoppingCartOpen
- SinWave
- Skull
- Soccer
- Sparkle
- Split
- Spoon
- SquareWave
- SteeringWheel
- SuitClub
- SuitDiamond
- SuitSpade
- Sword
- Tag
- TagLabel
- Telescope
- Thermometer
- Ticket
- TrafficCone
- Transform
- TriangleWave
- Trident
- Umbrella
- WeatherCloud
- WeatherFog
- WeatherLightning
- WeatherPartialCloudDay
- WeatherPartialCloudNight
- WeatherRain
- WeatherSnow
- WeatherWind
- Weight
- WeightBar
- Wheel
- Wheelchair
- WineGlass
- Woman
- Yoke

### Shapes

- ArrowDown
- ArrowDownLeft
- ArrowDownMax
- ArrowDownRight
- ArrowLeft
- ArrowLeftMax
- ArrowLeftRight
- ArrowRight
- ArrowRightMax
- ArrowUp
- ArrowUpDown
- ArrowUpLeft
- ArrowUpMax
- ArrowUpRight
- ChevronDoubleDown
- ChevronDoubleLeft
- ChevronDoubleRight
- ChevronDoubleUp
- ChevronDown
- ChevronDownLeftUpRight
- ChevronLeft
- ChevronLeftRight
- ChevronRight
- ChevronUp
- ChevronUpDown
- ChevronUpLeftDownRight
- Diamond
- DiamondOff
- Return
- ShapeCircle
- ShapeCircleOff
- ShapeHexagon
- ShapeOctagon
- ShapeSquare
- ShapeSquareOff
- ShapeTriangle

### Social

- AccountAdd
- AccountCheck
- AccountMore
- AccountRemove
- AddressBook
- AwardRibbon
- AwardTrophy
- Baby
- Bookmark
- BookmarkOff
- ChatMessage
- ChatMulti
- ChatNew
- ChatSingle
- Chatbot
- Family
- FeelingHappy
- FeelingNeutral
- FeelingSad
- FeelingVeryHappy
- FeelingVerySad
- Founders
- FoundersTier1
- FoundersTier2
- FoundersTier3
- Heart
- HeartOff
- Identification
- License
- Pin
- PinOff
- Profile
- ProfileGroup
- ProfileOff
- RankingXp
- ServiceAccount
- ThumbDown
- ThumbUp

### View

- Apps
- CarouselHorizontal
- CarouselVertical
- DisplayShare
- Dual
- Feedback
- FitToPage
- Fullscreen
- FullscreenExit
- GalleryStrip
- GripArea
- GripCorner
- GripEdgeHorizontal
- GripEdgeVertical
- GripMini
- Icon3SquarePlus
- Icon4Square
- IconWindow
- LayoutColumns
- LayoutDetail
- LayoutGrid
- LayoutList
- LayoutRows
- OpenExternal
- PillBarStack
- RectangleGroup
- Render
- Segmap
- StackHorizontal
- WindowCode
- WindowGrid
- WindowStack
- WindowTerminal
- Workspace
- WorkspaceGroup
