# Workspace Page (Layout Reference)

Use for AI-assisted, editor-like, or command-center pages where the user works inside a persistent task surface. Workspace pages are not dashboards or detail pages: their primary content is an active interaction area, usually a conversation, editor, run console, or mixed tool canvas.

## When to use

- The page centers on an ongoing task or conversation rather than browsing a collection
- Users need persistent context while they work: model details, run settings, files, history, citations, or output metadata
- The center region is interactive and may stream, update, or accept user input
- The route is a durable workspace, not a transient modal or side panel

**Examples:** AI chat workspace, model playground, run investigation console, notebook-like assistant surface, task triage workspace.

## Choose the right variant

| Variant                  | Use when                                                                              | Structure                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Three-pane workspace** | The conversation or task needs persistent left and right context while the user works | AppBar, full-width body, 340px left [Panel](../components/Panel.md), 1240px center workspace, 340px right [Panel](../components/Panel.md) |
| **Focused chat**         | Conversation is the whole task and side context would distract                        | AppBar, centered `max-width: 800px` chat column                                                                                           |

Default to focused chat when the surrounding panels do not contain information the user needs while composing the next message.

---

## Common structure

Both variants keep the standard sticky `AppBar` from [layout-principles.md](./layout-principles.md). Unlike list, detail, and dashboard pages, the workspace body may be full-width because the rails and center canvas define the layout.

### Three-pane workspace

- Left rail: [Panel](../components/Panel.md) for durable context such as model identity, tags, metadata, and quality signals.
- Center: workspace with [Tabs](../components/Tabs.md), a bordered center header, transcript or primary work surface, thinking/loading skeletons, chat input, and disclaimer.
- Right rail: [Panel](../components/Panel.md) for run settings, selected-output metadata, code snippets, or secondary controls.

Do not turn the rails into floating cards inside another card. The rails are the page regions.

### Focused chat

Use a centered `max-width: 800px` column when the conversation itself is the product:

---

## Interaction guidance

- Keep chat input visible whenever the page is ready for user input.
- Preserve transcript position while streaming or showing a thinking skeleton.
- Use [Tabs](../components/Tabs.md) for center workspace modes only when each tab changes the primary work surface.
- Keep side panels persistent only when their content changes the user's next action; otherwise collapse the information into the focused chat flow.

## React examples

### Chat input

Pinned footer: vertical `TextArea` with tertiary attach (`label` + `nv-sr-only` `input[type=file]`) and brand Send in `slotEnd`, plus a centered disclaimer line.

```tsx
<Stack gap="2">
  <TextArea
    layout="vertical"
    placeholder="Ask anything"
    resizeable="auto"
    slotStart={<UploadedItems />}
    slotEnd={
      <Flex className="w-full" gap="inherit" align="center" justify="between">
        <Dropdown items={["Instant", "Thinking", "Research"]} size="small">
          Instant
        </Dropdown>
        <Flex gap="inherit">
          <Button kind="tertiary" size="small" asChild>
            <label aria-label="Attach file">
              <input className="nv-sr-only" type="file" data-hidden="true" />
              <Icon name="paperclip" aria-hidden />
            </label>
          </Button>
          <Button color="brand" size="small" aria-label="Send">
            <Icon name="paperplane" variant="fill" aria-hidden />
          </Button>
        </Flex>
      </Flex>
    }
  />
  <Text kind="label/regular/sm" className="text-center text-placeholder">
    AI can make mistakes.
  </Text>
</Stack>
```

### Focused chat

Workspace focused-chat variant: stacked shell from layout-principles.md, then a centered `max-w-[800px]` column with transcript and fixed bottom input.

```tsx
<>
  {/* Page shell: layout-principles.md (full-width <main>).
	    Focused-chat variant: centered 800px column inside <main>. */}
  <div className="mx-auto flex w-full max-w-[800px] flex-col gap-6">
    <Stack gap="6">
      {messages.map((message) => {
        if (message.role === "user") {
          // user messages are right aligned, max width 700px to promote readability
          return (
            <Stack gap="2" align="end" key={message.id}>
              <div className="max-w-[700px] rounded-tl-xl rounded-tr-xl rounded-bl-xl bg-background-subtle p-4">
                <Text kind="body/regular/lg">{message.content}</Text>
              </div>
              {message.meta && (
                <Text kind="label/regular/sm" className="text-secondary">
                  {message.meta}
                </Text>
              )}
            </Stack>
          );
        }
        // system messages are left aligned, with 40px padding from right edge
        return (
          <Stack gap="2" className="pr-10" key={message.id}>
            {message.role === "thinking" ? (
              <>
                <Text kind="body/regular/lg" className="text-placeholder">
                  Thinking...
                </Text>
                <Skeleton kind="pill" style={{ height: 8, width: "100%" }} />
                <Skeleton kind="pill" style={{ height: 8, width: "100%" }} />
                <Skeleton kind="pill" style={{ height: 8, width: "52%" }} />
              </>
            ) : (
              <>
                <Text kind="body/regular/lg">{message.content}</Text>
                {message.meta && (
                  <Text kind="label/regular/sm" className="text-secondary">
                    {message.meta}
                  </Text>
                )}
              </>
            )}
          </Stack>
        );
      })}
    </Stack>
    <Stack gap="2">
      <TextArea
        layout="vertical"
        placeholder="Ask anything"
        resizeable="auto"
        slotStart={<UploadedItems />}
        slotEnd={
          <Flex className="w-full" gap="inherit" align="center" justify="between">
            <Dropdown items={["Instant", "Thinking", "Research"]} size="small">
              Instant
            </Dropdown>
            <Flex gap="inherit">
              <Button kind="tertiary" size="small" asChild>
                <label aria-label="Attach file">
                  <input className="nv-sr-only" type="file" data-hidden="true" />
                  <Icon name="paperclip" aria-hidden />
                </label>
              </Button>
              <Button color="brand" size="small" aria-label="Send">
                <Icon name="paperplane" variant="fill" aria-hidden />
              </Button>
            </Flex>
          </Flex>
        }
      />
      <Text kind="label/regular/sm" className="text-center text-placeholder">
        AI can make mistakes.
      </Text>
    </Stack>
  </div>
</>
```

### Three-pane workspace

Workspace three-pane variant (see workspace-page.md): stacked shell from layout-principles.md, then a viewport-height, overflow-hidden grid with a fluid left Panel, 1240px center workspace (tabs, bordered header, transcript, thinking skeleton, input, disclaimer), and a fluid right Panel. Give each visible pane its own vertical scrolling so the document does not become the workspace scroller. Side Panels are `clamp(280px,24vw,340px)` — capped at 340px so on ultrawide screens they hug the centered workspace with empty space at the edges, and shrinking toward a 280px floor as the viewport narrows. Below the `lg` breakpoint they are hidden and the center workspace goes full-width.

```tsx
<>
  {/* Page shell: layout-principles.md (full-width <main>).
	    Three-pane workspace variant — center column is the layout container. */}
  <Grid className="h-screen min-h-0 grid-cols-1 justify-center overflow-hidden lg:grid-cols-[clamp(280px,24vw,340px)_minmax(0,1240px)_clamp(280px,24vw,340px)]">
    <Panel className="hidden h-full overflow-y-auto rounded-none border-y-none border-r-1 border-l-none lg:flex">
      <Stack gap="4">
        <PageHeader
          kind="flat"
          slotSubheading={
            <Text className="text-secondary" kind="body/regular/md">
              <Flex align="center" gap="1">
                <LogoIcon /> NVIDIA
              </Flex>
            </Text>
          }
          slotHeading={
            <Flex gap="2" align="center">
              <Text kind="body/bold/2xl">Nemotron Agent</Text>
              <Badge color="green" kind="solid">
                Ready
              </Badge>
            </Flex>
          }
          slotDescription={
            <Text className="text-secondary" kind="body/regular/md">
              Lorem ipsum dolor sit amet
            </Text>
          }
        />
        <Flex gap="2" wrap="wrap">
          <Tag color="gray" kind="outline" readOnly>
            Tag
          </Tag>
          <Tag color="gray" kind="outline" readOnly>
            Tag
          </Tag>
          <Tag color="gray" kind="outline" readOnly>
            Tag
          </Tag>
        </Flex>
        <Divider />
        <Stack className="text-secondary" gap="4">
          <Text kind="body/bold/md">Model Details</Text>
          <Stack className="pb-4">
            {modelKv.map((kv) => (
              <Flex key={kv.label} className="border-base border-b py-1" align="center" justify="between" gap="1">
                <Text kind="body/regular/sm" className="text-secondary">
                  {kv.label}
                </Text>
                <Text className="text-primary" kind="body/semibold/md">
                  {kv.value}
                </Text>
              </Flex>
            ))}
          </Stack>
          <Text kind="body/bold/md">Quality</Text>
          {["Grounding", "Reasoning", "Retrieval", "Safety"].map((label) => (
            <Stack key={label} gap="1">
              <Flex justify="between">
                <Text kind="label/regular/sm">{label}</Text>
                <Text kind="label/regular/sm">88.4%</Text>
              </Flex>
              <ProgressBar aria-label={`${label} score`} value={88} size="small" />
            </Stack>
          ))}
        </Stack>
      </Stack>
    </Panel>

    <section className="h-full min-h-0 w-full max-w-7xl overflow-y-auto">
      <Stack>
        <Tabs
          defaultValue="chat"
          items={[
            { value: "chat", children: "Chat" },
            { value: "context", children: "Context" },
            { value: "runs", children: "Runs" },
            { value: "files", children: "Files" },
            { value: "notes", children: "Notes" },
          ]}
        />
        <PageHeader
          className="border-base border-b px-6 py-4"
          kind="flat"
          slotHeading={
            <Flex gap="2" align="center" justify="between">
              <Text kind="body/bold/xl">Lorem Ipsum</Text>
              <Dropdown items={["Action 1", "Action 2", "Action 3"]} size="small">
                <Icon name="more-vert" variant="fill" aria-label="More actions" />
              </Dropdown>
            </Flex>
          }
        />
        <div className="flex flex-1 flex-col gap-6 px-[clamp(2px,5vw,112px)] pt-8 pb-3">
          <>
            {/* Page shell: layout-principles.md (full-width <main>).
						    Focused-chat variant: centered 800px column inside <main>. */}
            <div className="mx-auto flex w-full max-w-[800px] flex-col gap-6">
              <Stack gap="6">
                {messages.map((message) => {
                  if (message.role === "user") {
                    // user messages are right aligned, max width 700px to promote readability
                    return (
                      <Stack gap="2" align="end" key={message.id}>
                        <div className="max-w-[700px] rounded-tl-xl rounded-tr-xl rounded-bl-xl bg-background-subtle p-4">
                          <Text kind="body/regular/lg">{message.content}</Text>
                        </div>
                        {message.meta && (
                          <Text kind="label/regular/sm" className="text-secondary">
                            {message.meta}
                          </Text>
                        )}
                      </Stack>
                    );
                  }
                  // system messages are left aligned, with 40px padding from right edge
                  return (
                    <Stack gap="2" className="pr-10" key={message.id}>
                      {message.role === "thinking" ? (
                        <>
                          <Text kind="body/regular/lg" className="text-placeholder">
                            Thinking...
                          </Text>
                          <Skeleton kind="pill" style={{ height: 8, width: "100%" }} />
                          <Skeleton kind="pill" style={{ height: 8, width: "100%" }} />
                          <Skeleton kind="pill" style={{ height: 8, width: "52%" }} />
                        </>
                      ) : (
                        <>
                          <Text kind="body/regular/lg">{message.content}</Text>
                          {message.meta && (
                            <Text kind="label/regular/sm" className="text-secondary">
                              {message.meta}
                            </Text>
                          )}
                        </>
                      )}
                    </Stack>
                  );
                })}
              </Stack>
              <Stack gap="2">
                <TextArea
                  layout="vertical"
                  placeholder="Ask anything"
                  resizeable="auto"
                  slotStart={<UploadedItems />}
                  slotEnd={
                    <Flex className="w-full" gap="inherit" align="center" justify="between">
                      <Dropdown items={["Instant", "Thinking", "Research"]} size="small">
                        Instant
                      </Dropdown>
                      <Flex gap="inherit">
                        <Button kind="tertiary" size="small" asChild>
                          <label aria-label="Attach file">
                            <input className="nv-sr-only" type="file" data-hidden="true" />
                            <Icon name="paperclip" aria-hidden />
                          </label>
                        </Button>
                        <Button color="brand" size="small" aria-label="Send">
                          <Icon name="paperplane" variant="fill" aria-hidden />
                        </Button>
                      </Flex>
                    </Flex>
                  }
                />
                <Text kind="label/regular/sm" className="text-center text-placeholder">
                  AI can make mistakes.
                </Text>
              </Stack>
            </div>
          </>
        </div>
      </Stack>
    </section>

    <Panel className="hidden h-full overflow-y-auto rounded-none border-y-none border-r-none border-l-1 lg:flex">
      <Stack className="text-secondary" gap="6">
        <Stack gap="3">
          <Text kind="body/bold/md">Run settings</Text>
          {runKv.map((kv) => (
            <Flex key={kv.label} className="border-base border-b py-1" justify="between">
              <Text kind="label/regular/sm">{kv.label}</Text>
              <Text className="text-primary" kind="body/semibold/md">
                {kv.value}
              </Text>
            </Flex>
          ))}
        </Stack>
        <Stack gap="3">
          <Text kind="body/bold/md">Code</Text>
          <CodeSnippet copyButton="inline" value={"Lorem ipsum dolor sit amet, consectetur adipiscing elit."} />
        </Stack>
      </Stack>
    </Panel>
  </Grid>
</>
```
