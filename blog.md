---
layout: page
title: Research Notes
permalink: /blog/
---

This page collects my paper reading notes and technical thoughts.

{% for post in site.posts %}
- **{{ post.date | date: "%Y-%m-%d" }}**  
  [{{ post.title }}]({{ post.url }})
{% endfor %}
