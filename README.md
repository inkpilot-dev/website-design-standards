# website-design-standards

A Claude Code plugin with one skill. Whenever Claude designs or builds a website, the skill loads and applies a checklist of anti-vibe-coded rules covering colour, layout, icons and imagery, typography, components, compliance (Terms and Privacy pages) and copy. Claude checks the finished site against every rule before handing it over.

## Install locally

```
/plugin marketplace add ./website-design-standards
/plugin install website-design-standards@website-design-standards
```

Or push this folder to a GitHub repo and run `/plugin marketplace add <owner>/<repo>`.

No setup is needed after install. Edit `skills/website-design-standards/SKILL.md` to change the rules.
